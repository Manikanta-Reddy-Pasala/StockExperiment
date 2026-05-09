# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1198.00
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 9 |
| TARGET_HIT | 17 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 4
- **Target hits / Stop hits / Partials:** 13 / 4 / 9
- **Avg / median % per leg:** 6.36% / 8.84%
- **Sum % (uncompounded):** 165.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 4 | 3 | 0 | 4.67% | 32.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 4 | 3 | 0 | 4.67% | 32.7% |
| SELL (all) | 19 | 18 | 94.7% | 9 | 1 | 9 | 6.98% | 132.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 18 | 94.7% | 9 | 1 | 9 | 6.98% | 132.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 22 | 84.6% | 13 | 4 | 9 | 6.36% | 165.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:45:00 | 1395.90 | 1395.18 | 1457.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 1432.60 | 1399.27 | 1454.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 1465.70 | 1405.31 | 1454.03 | SL hit (close>static) qty=1.00 sl=1464.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1430.00 | 1406.19 | 1453.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:00:00 | 1433.70 | 1406.19 | 1453.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1421.80 | 1408.19 | 1453.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1455.00 | 1409.86 | 1452.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:45:00 | 1440.10 | 1410.96 | 1452.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1441.60 | 1412.34 | 1452.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 1443.60 | 1412.67 | 1452.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 1436.70 | 1414.53 | 1452.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1431.30 | 1415.99 | 1451.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 1424.40 | 1416.86 | 1451.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1308.00 | 1418.57 | 1450.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1358.50 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1362.01 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1350.71 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1368.09 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1369.52 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1371.42 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1364.87 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 1353.18 | 1417.45 | 1449.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-22 10:15:00 | 1299.24 | 1416.27 | 1449.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1290.33 | 1415.00 | 1448.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1296.09 | 1415.00 | 1448.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1297.44 | 1415.00 | 1448.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 11:15:00 | 1293.03 | 1415.00 | 1448.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1287.00 | 1413.72 | 1447.63 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-22 12:15:00 | 1281.96 | 1413.72 | 1447.63 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-04-23 09:15:00 | 1279.62 | 1408.61 | 1444.38 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1242.60 | 1399.23 | 1438.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-05-08 09:15:00 | 1177.20 | 1306.02 | 1374.12 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-05 12:15:00 | 1608.90 | 2026-02-03 09:15:00 | 1769.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 13:15:00 | 1608.30 | 2026-02-03 09:15:00 | 1769.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 14:45:00 | 1610.10 | 2026-02-03 09:15:00 | 1771.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1611.90 | 2026-02-03 09:15:00 | 1773.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1636.60 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1642.00 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-05 09:45:00 | 1625.10 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-04-08 10:15:00 | 1432.60 | 2026-04-09 14:15:00 | 1465.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 09:15:00 | 1358.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 09:15:00 | 1362.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 09:15:00 | 1350.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 09:15:00 | 1368.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 09:15:00 | 1369.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 09:15:00 | 1371.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 09:15:00 | 1364.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-22 09:15:00 | 1353.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 10:15:00 | 1299.24 | TARGET_HIT | 0.50 | 9.14% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 11:15:00 | 1290.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 11:15:00 | 1296.09 | TARGET_HIT | 0.50 | 8.84% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 11:15:00 | 1297.44 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 11:15:00 | 1293.03 | TARGET_HIT | 0.50 | 10.31% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 12:15:00 | 1287.00 | TARGET_HIT | 0.50 | 10.85% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 12:15:00 | 1281.96 | TARGET_HIT | 0.50 | 10.77% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-23 09:15:00 | 1279.62 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-04-24 09:15:00 | 1242.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-05-08 09:15:00 | 1177.20 | TARGET_HIT | 0.50 | 10.00% |
