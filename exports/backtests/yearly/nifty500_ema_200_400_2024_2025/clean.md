# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 891.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 28
- **Target hits / Stop hits / Partials:** 1 / 29 / 2
- **Avg / median % per leg:** -3.00% / -2.92%
- **Sum % (uncompounded):** -95.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.82% | -45.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.82% | -45.2% |
| SELL (all) | 16 | 4 | 25.0% | 1 | 13 | 2 | -3.17% | -50.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 1 | 13 | 2 | -3.17% | -50.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 4 | 12.5% | 1 | 29 | 2 | -3.00% | -95.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 1457.05 | 1347.56 | 1347.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 13:15:00 | 1472.35 | 1361.24 | 1354.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1437.75 | 1442.36 | 1409.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 1437.75 | 1442.36 | 1409.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1428.00 | 1443.36 | 1412.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1425.25 | 1443.36 | 1412.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1483.70 | 1545.78 | 1505.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1483.70 | 1545.78 | 1505.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1472.30 | 1545.04 | 1504.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1472.30 | 1545.04 | 1504.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1484.60 | 1541.25 | 1504.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1484.60 | 1541.25 | 1504.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1481.75 | 1537.54 | 1503.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 1481.75 | 1537.54 | 1503.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1508.00 | 1529.80 | 1502.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:15:00 | 1513.50 | 1529.80 | 1502.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:45:00 | 1514.25 | 1529.62 | 1502.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1519.85 | 1529.21 | 1502.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 1501.00 | 1528.76 | 1502.35 | SL hit (close<static) qty=1.00 sl=1502.05 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1465.00 | 1532.78 | 1532.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1435.95 | 1529.54 | 1531.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 11:15:00 | 1378.50 | 1359.73 | 1419.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:00:00 | 1378.50 | 1359.73 | 1419.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1407.55 | 1361.24 | 1418.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 14:00:00 | 1373.10 | 1427.38 | 1432.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 15:15:00 | 1368.05 | 1426.92 | 1432.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:45:00 | 1374.80 | 1425.80 | 1431.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:00:00 | 1375.80 | 1424.87 | 1431.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1425.60 | 1417.66 | 1426.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 1428.25 | 1417.66 | 1426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1425.00 | 1417.74 | 1426.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1436.45 | 1419.32 | 1426.72 | SL hit (close>static) qty=1.00 sl=1436.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1425.20 | 1255.50 | 1254.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1441.50 | 1271.93 | 1263.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1407.20 | 1412.68 | 1360.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1407.20 | 1412.68 | 1360.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1350.60 | 1444.01 | 1407.19 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 1233.90 | 1379.04 | 1379.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 1224.40 | 1357.26 | 1367.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 1195.70 | 1192.79 | 1239.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:30:00 | 1200.90 | 1192.79 | 1239.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 747.30 | 721.12 | 757.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 751.40 | 721.12 | 757.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 752.00 | 722.94 | 757.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 746.30 | 723.45 | 757.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 747.85 | 724.15 | 757.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 742.40 | 727.02 | 757.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 745.15 | 727.72 | 757.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 755.85 | 728.69 | 757.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 758.25 | 728.69 | 757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 753.90 | 729.44 | 757.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 757.35 | 729.44 | 757.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 759.00 | 730.58 | 757.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 759.00 | 730.58 | 757.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 766.45 | 730.93 | 757.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:30:00 | 770.05 | 730.93 | 757.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 764.05 | 731.26 | 757.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 831.95 | 732.94 | 757.56 | SL hit (close>static) qty=1.00 sl=770.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 875.60 | 775.43 | 775.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 879.00 | 779.40 | 777.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-04 13:15:00 | 1513.50 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-04 13:45:00 | 1514.25 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1519.85 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1514.90 | 2024-09-06 09:15:00 | 1488.35 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-09-09 11:15:00 | 1529.00 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-09-09 12:30:00 | 1524.55 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-09-19 11:30:00 | 1525.90 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-09-24 14:30:00 | 1524.90 | 2024-10-07 14:15:00 | 1506.95 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-25 09:15:00 | 1526.10 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-09-25 14:45:00 | 1532.25 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-09-26 11:00:00 | 1529.25 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2024-10-07 12:30:00 | 1532.00 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-11-06 11:45:00 | 1542.95 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-11-06 14:15:00 | 1542.70 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-11-06 14:45:00 | 1541.95 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2024-11-07 10:15:00 | 1542.30 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-01-13 14:00:00 | 1373.10 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-01-13 15:15:00 | 1368.05 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-01-14 09:45:00 | 1374.80 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-01-14 12:00:00 | 1375.80 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2025-01-22 09:15:00 | 1419.20 | 2025-01-23 09:15:00 | 1439.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-22 11:15:00 | 1422.85 | 2025-01-23 09:15:00 | 1439.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1416.15 | 2025-01-27 09:15:00 | 1345.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1416.15 | 2025-01-31 09:15:00 | 1404.75 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2025-02-01 11:30:00 | 1423.10 | 2025-02-01 14:15:00 | 1438.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-02-01 13:15:00 | 1397.90 | 2025-02-01 14:15:00 | 1438.65 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1393.75 | 2025-02-12 09:15:00 | 1324.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1393.75 | 2025-02-27 13:15:00 | 1254.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 11:45:00 | 746.30 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.48% |
| SELL | retest2 | 2026-04-16 14:30:00 | 747.85 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.25% |
| SELL | retest2 | 2026-04-20 09:30:00 | 742.40 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -12.06% |
| SELL | retest2 | 2026-04-20 13:15:00 | 745.15 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.65% |
