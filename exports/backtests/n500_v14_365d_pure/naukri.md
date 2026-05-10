# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 978.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 27
- **Target hits / Stop hits / Partials:** 1 / 32 / 6
- **Avg / median % per leg:** -0.42% / -1.06%
- **Sum % (uncompounded):** -16.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 12 | 30.8% | 1 | 32 | 6 | -0.42% | -16.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 12 | 30.8% | 1 | 32 | 6 | -0.42% | -16.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 12 | 30.8% | 1 | 32 | 6 | -0.42% | -16.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 14:15:00)

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

### Cycle 2 — SELL (started 2025-07-15 13:15:00)

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
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1388.20 | 1373.95 | 1393.00 | SL hit (close>ema200) qty=0.50 sl=1373.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1388.20 | 1373.95 | 1393.00 | SL hit (close>ema200) qty=0.50 sl=1373.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1388.20 | 1373.95 | 1393.00 | SL hit (close>ema200) qty=0.50 sl=1373.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1383.20 | 1374.13 | 1392.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1393.50 | 1374.32 | 1393.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:15:00 | 1390.00 | 1374.32 | 1393.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1380.50 | 1374.38 | 1392.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 1378.70 | 1374.43 | 1392.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1378.80 | 1374.43 | 1392.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1371.90 | 1374.63 | 1392.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 1377.20 | 1371.73 | 1388.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1383.50 | 1372.08 | 1388.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 1381.10 | 1372.08 | 1388.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1378.50 | 1372.15 | 1388.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 1388.30 | 1372.15 | 1388.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1396.00 | 1373.02 | 1387.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1396.80 | 1373.02 | 1387.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1397.00 | 1373.25 | 1387.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1397.00 | 1373.25 | 1387.80 | SL hit (close>static) qty=1.00 sl=1396.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1397.00 | 1373.25 | 1387.80 | SL hit (close>static) qty=1.00 sl=1396.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1397.00 | 1373.25 | 1387.80 | SL hit (close>static) qty=1.00 sl=1396.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1397.00 | 1373.25 | 1387.80 | SL hit (close>static) qty=1.00 sl=1396.70 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1402.00 | 1373.25 | 1387.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1414.40 | 1373.66 | 1387.94 | SL hit (close>static) qty=1.00 sl=1411.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1414.40 | 1373.66 | 1387.94 | SL hit (close>static) qty=1.00 sl=1411.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1414.40 | 1373.66 | 1387.94 | SL hit (close>static) qty=1.00 sl=1411.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1414.40 | 1373.66 | 1387.94 | SL hit (close>static) qty=1.00 sl=1411.20 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1398.40 | 1377.20 | 1388.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 1395.90 | 1377.20 | 1388.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1396.30 | 1377.39 | 1388.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 1401.00 | 1377.39 | 1388.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1378.60 | 1361.13 | 1377.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 1378.60 | 1361.13 | 1377.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1380.60 | 1361.33 | 1377.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:30:00 | 1384.00 | 1361.33 | 1377.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1386.30 | 1361.92 | 1377.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 1385.00 | 1361.92 | 1377.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1385.30 | 1362.15 | 1377.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1385.90 | 1362.15 | 1377.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1349.80 | 1363.31 | 1377.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 1343.80 | 1363.31 | 1377.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 11:30:00 | 1343.20 | 1362.99 | 1376.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:45:00 | 1342.50 | 1362.77 | 1376.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 1341.80 | 1362.56 | 1376.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1368.60 | 1350.32 | 1366.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 1368.60 | 1350.32 | 1366.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1376.40 | 1350.58 | 1366.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1383.00 | 1350.58 | 1366.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1377.00 | 1351.13 | 1366.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 1383.90 | 1351.13 | 1366.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 1409.50 | 1352.04 | 1366.58 | SL hit (close>static) qty=1.00 sl=1393.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 1409.50 | 1352.04 | 1366.58 | SL hit (close>static) qty=1.00 sl=1393.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 1409.50 | 1352.04 | 1366.58 | SL hit (close>static) qty=1.00 sl=1393.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 1409.50 | 1352.04 | 1366.58 | SL hit (close>static) qty=1.00 sl=1393.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1368.00 | 1354.52 | 1367.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 1368.00 | 1354.52 | 1367.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1367.10 | 1354.64 | 1367.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 1367.10 | 1354.64 | 1367.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1361.00 | 1354.71 | 1367.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 1367.50 | 1354.71 | 1367.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1364.10 | 1354.80 | 1367.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:30:00 | 1368.70 | 1354.80 | 1367.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1366.40 | 1354.92 | 1367.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1388.20 | 1354.92 | 1367.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1374.00 | 1355.11 | 1367.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1358.10 | 1362.13 | 1369.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1348.90 | 1354.28 | 1363.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 1365.80 | 1349.50 | 1359.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 1367.00 | 1349.99 | 1359.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1348.40 | 1349.94 | 1358.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 1338.00 | 1350.05 | 1358.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 1340.50 | 1349.89 | 1358.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 1339.40 | 1349.75 | 1358.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:00:00 | 1340.50 | 1348.93 | 1357.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1368.00 | 1346.42 | 1355.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1368.00 | 1346.42 | 1355.29 | SL hit (close>static) qty=1.00 sl=1365.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1368.00 | 1346.42 | 1355.29 | SL hit (close>static) qty=1.00 sl=1365.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1368.00 | 1346.42 | 1355.29 | SL hit (close>static) qty=1.00 sl=1365.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1368.00 | 1346.42 | 1355.29 | SL hit (close>static) qty=1.00 sl=1365.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 1368.00 | 1346.42 | 1355.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1392.10 | 1347.32 | 1355.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1392.10 | 1347.32 | 1355.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1397.80 | 1351.77 | 1357.35 | SL hit (close>static) qty=1.00 sl=1393.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1397.80 | 1351.77 | 1357.35 | SL hit (close>static) qty=1.00 sl=1393.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1397.80 | 1351.77 | 1357.35 | SL hit (close>static) qty=1.00 sl=1393.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1397.80 | 1351.77 | 1357.35 | SL hit (close>static) qty=1.00 sl=1393.20 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1359.60 | 1357.24 | 1359.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:15:00 | 1360.00 | 1357.24 | 1359.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1371.30 | 1357.38 | 1359.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1371.30 | 1357.38 | 1359.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1370.20 | 1357.51 | 1359.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 1374.80 | 1357.51 | 1359.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1366.80 | 1358.29 | 1360.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 1366.80 | 1358.29 | 1360.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1358.40 | 1356.78 | 1359.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 1357.20 | 1356.78 | 1359.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1364.20 | 1356.86 | 1359.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 1364.20 | 1356.86 | 1359.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1366.90 | 1356.96 | 1359.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 1366.90 | 1356.96 | 1359.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1360.90 | 1357.41 | 1359.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 1360.70 | 1357.41 | 1359.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1360.90 | 1357.45 | 1359.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 1360.90 | 1357.45 | 1359.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1360.00 | 1357.47 | 1359.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1351.30 | 1357.47 | 1359.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 1353.40 | 1357.49 | 1359.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:00:00 | 1351.80 | 1356.82 | 1359.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 1358.40 | 1353.04 | 1356.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1359.30 | 1353.10 | 1356.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 1360.00 | 1353.10 | 1356.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1359.00 | 1353.16 | 1356.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:45:00 | 1359.10 | 1353.16 | 1356.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1367.70 | 1353.30 | 1356.79 | SL hit (close>static) qty=1.00 sl=1362.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1367.70 | 1353.30 | 1356.79 | SL hit (close>static) qty=1.00 sl=1362.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1367.70 | 1353.30 | 1356.79 | SL hit (close>static) qty=1.00 sl=1362.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1367.70 | 1353.30 | 1356.79 | SL hit (close>static) qty=1.00 sl=1362.80 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1353.80 | 1355.77 | 1357.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 1352.60 | 1355.77 | 1357.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1359.40 | 1355.81 | 1357.76 | SL hit (close>static) qty=1.00 sl=1358.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1346.60 | 1355.83 | 1357.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1362.00 | 1352.51 | 1355.79 | SL hit (close>static) qty=1.00 sl=1358.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1348.00 | 1353.21 | 1356.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1374.30 | 1352.02 | 1355.23 | SL hit (close>static) qty=1.00 sl=1358.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1351.00 | 1352.70 | 1355.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1283.45 | 1337.94 | 1346.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 1339.60 | 1337.02 | 1345.93 | SL hit (close>ema200) qty=0.50 sl=1337.02 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1337.40 | 1336.82 | 1345.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 1326.80 | 1336.82 | 1345.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 1260.46 | 1330.98 | 1341.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-04 09:15:00 | 1194.12 | 1303.80 | 1324.75 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
