# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1268.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -2.75% / -2.06%
- **Sum % (uncompounded):** -22.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.75% | -22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.75% | -22.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.75% | -22.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 1256.60 | 1083.58 | 1082.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 1274.30 | 1089.03 | 1085.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1525.10 | 1525.92 | 1435.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 1525.10 | 1525.92 | 1435.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1441.40 | 1518.48 | 1440.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1441.40 | 1518.48 | 1440.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1454.80 | 1517.84 | 1440.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:45:00 | 1457.60 | 1517.19 | 1440.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1429.50 | 1513.48 | 1440.56 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 1462.80 | 1504.28 | 1439.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 1458.90 | 1502.77 | 1439.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 1459.40 | 1501.86 | 1440.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1447.00 | 1500.37 | 1440.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 1466.50 | 1500.37 | 1440.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1453.40 | 1499.91 | 1440.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 1477.00 | 1498.87 | 1440.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1432.60 | 1497.13 | 1440.97 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1432.60 | 1497.13 | 1440.97 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1432.60 | 1497.13 | 1440.97 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1419.70 | 1493.92 | 1440.73 | SL hit (close<static) qty=1.00 sl=1425.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 1473.90 | 1455.30 | 1432.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 1475.50 | 1457.54 | 1434.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 11:15:00 | 1474.80 | 1457.54 | 1434.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1441.10 | 1457.18 | 1434.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 1437.80 | 1457.18 | 1434.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1444.90 | 1457.06 | 1434.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1436.00 | 1457.06 | 1434.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1443.00 | 1456.92 | 1434.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 1424.00 | 1456.92 | 1434.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1472.00 | 1498.82 | 1470.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 1474.40 | 1498.82 | 1470.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1468.00 | 1498.51 | 1470.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1468.00 | 1498.51 | 1470.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1482.60 | 1498.35 | 1470.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1477.70 | 1498.35 | 1470.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1469.20 | 1497.67 | 1470.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1472.00 | 1497.67 | 1470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1470.00 | 1497.40 | 1470.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1467.70 | 1497.40 | 1470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1470.00 | 1497.13 | 1470.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 1470.00 | 1497.13 | 1470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1471.40 | 1496.87 | 1470.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:30:00 | 1469.50 | 1496.87 | 1470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1469.30 | 1496.60 | 1470.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1469.30 | 1496.60 | 1470.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1460.00 | 1496.23 | 1470.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 1460.00 | 1496.23 | 1470.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1460.00 | 1490.98 | 1468.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 1464.30 | 1490.98 | 1468.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1463.20 | 1490.70 | 1468.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:45:00 | 1463.00 | 1490.70 | 1468.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1461.20 | 1490.41 | 1468.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1461.20 | 1490.41 | 1468.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1465.00 | 1490.16 | 1468.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1475.90 | 1490.16 | 1468.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1455.70 | 1489.59 | 1468.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 1455.50 | 1489.59 | 1468.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 1423.00 | 1480.03 | 1465.85 | SL hit (close<static) qty=1.00 sl=1425.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 1423.00 | 1480.03 | 1465.85 | SL hit (close<static) qty=1.00 sl=1425.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 1423.00 | 1480.03 | 1465.85 | SL hit (close<static) qty=1.00 sl=1425.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-09-26 11:15:00 | 1309.90 | 1454.30 | 1454.41 | min_gap filter: gap=0.008% < 0.030% |
| TREND_RESET | 2025-09-26 11:15:00 | 1309.90 | 1454.30 | 1454.41 | EMA inversion without crossover edge (EMA200=1454.30 EMA400=1454.41) — end cycle |
| CROSSOVER_SKIP | 2026-03-19 09:15:00 | 1117.80 | 1100.19 | 1100.18 | min_gap filter: gap=0.001% < 0.030% |
| CROSSOVER_SKIP | 2026-03-23 09:15:00 | 1033.70 | 1100.18 | 1100.20 | min_gap filter: gap=0.002% < 0.030% |
| CROSSOVER_SKIP | 2026-04-16 15:15:00 | 1205.00 | 1092.67 | 1092.55 | min_gap filter: gap=0.010% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-25 12:45:00 | 1457.60 | 2025-07-28 10:15:00 | 1429.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-29 15:00:00 | 1462.80 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1458.90 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-30 12:30:00 | 1459.40 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-31 13:00:00 | 1477.00 | 2025-08-01 14:15:00 | 1419.70 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-08-14 11:00:00 | 1473.90 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-18 10:30:00 | 1475.50 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-08-18 11:15:00 | 1474.80 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.51% |
