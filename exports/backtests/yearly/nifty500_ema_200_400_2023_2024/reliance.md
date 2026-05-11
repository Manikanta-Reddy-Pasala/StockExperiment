# Reliance Industries Ltd. (RELIANCE)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1436.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 37 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 22 / 23
- **Target hits / Stop hits / Partials:** 9 / 30 / 6
- **Avg / median % per leg:** 1.92% / -0.45%
- **Sum % (uncompounded):** 86.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 8 | 34.8% | 5 | 18 | 0 | 0.92% | 21.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.06% | -9.2% |
| BUY @ 3rd Alert (retest2) | 20 | 8 | 40.0% | 5 | 15 | 0 | 1.51% | 30.3% |
| SELL (all) | 22 | 14 | 63.6% | 4 | 12 | 6 | 2.97% | 65.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 14 | 63.6% | 4 | 12 | 6 | 2.97% | 65.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.06% | -9.2% |
| retest2 (combined) | 42 | 22 | 52.4% | 9 | 27 | 6 | 2.28% | 95.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 12:15:00 | 1213.68 | 1262.14 | 1262.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 13:15:00 | 1210.85 | 1261.63 | 1261.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 1167.10 | 1164.90 | 1187.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-06 10:00:00 | 1167.10 | 1164.90 | 1187.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 1184.53 | 1164.81 | 1182.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 11:00:00 | 1184.53 | 1164.81 | 1182.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 1183.85 | 1164.99 | 1182.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 15:00:00 | 1177.05 | 1165.50 | 1182.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 14:15:00 | 1180.90 | 1166.51 | 1182.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 09:15:00 | 1186.25 | 1167.32 | 1182.21 | SL hit (close>static) qty=1.00 sl=1184.80 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 11:15:00 | 1232.70 | 1190.73 | 1190.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 1239.08 | 1200.33 | 1195.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 1453.58 | 1457.80 | 1407.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 11:45:00 | 1453.83 | 1457.80 | 1407.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1460.50 | 1465.32 | 1442.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:00:00 | 1463.28 | 1464.99 | 1442.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 1462.28 | 1464.87 | 1442.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 1461.15 | 1464.69 | 1442.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 1464.53 | 1464.25 | 1443.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1438.55 | 1464.93 | 1446.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1438.55 | 1464.93 | 1446.00 | SL hit (close<static) qty=1.00 sl=1441.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.98 | 1494.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1452.93 | 1493.57 | 1493.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.48 | 1484.59 | 1488.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 1487.48 | 1484.59 | 1488.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1489.55 | 1484.64 | 1488.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 1489.88 | 1484.64 | 1488.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 1492.50 | 1484.72 | 1488.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 1490.00 | 1484.72 | 1488.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1484.23 | 1484.71 | 1488.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 1470.78 | 1488.10 | 1489.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1397.24 | 1481.67 | 1486.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-24 10:15:00 | 1323.70 | 1407.43 | 1439.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.00 | 1249.14 | 1248.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.30 | 1249.67 | 1249.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.30 | 1439.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 12:00:00 | 1482.80 | 1481.30 | 1439.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:00:00 | 1485.00 | 1481.33 | 1439.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1481.90 | 1481.41 | 1440.80 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.53 | 1442.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1443.10 | 1480.53 | 1442.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1445.50 | 1480.18 | 1442.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1444.90 | 1480.18 | 1442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.90 | SL hit (close<ema400) qty=1.00 sl=1442.90 alert=retest1 |

### Cycle 5 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.44 | 1423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1380.50 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 1415.70 | 1413.28 | 1417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1419.50 | 1413.36 | 1417.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1419.50 | 1413.36 | 1417.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1421.00 | 1413.44 | 1417.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1412.50 | 1413.44 | 1417.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 1422.90 | 1413.60 | 1418.02 | SL hit (close>static) qty=1.00 sl=1421.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.01 | 1398.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1483.00 | 1401.36 | 1400.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1511.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1539.30 | 1541.17 | 1511.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1520.74 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1477.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 1459.10 | 1449.75 | 1471.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1462.00 | 1450.57 | 1468.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1462.60 | 1451.01 | 1468.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 1462.20 | 1452.02 | 1468.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1388.90 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.47 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1389.09 | 1435.62 | 1453.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1315.80 | 1427.64 | 1448.33 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-16 15:00:00 | 1177.05 | 2023-11-21 09:15:00 | 1186.25 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-11-17 14:15:00 | 1180.90 | 2023-11-21 09:15:00 | 1186.25 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-04-25 14:00:00 | 1463.28 | 2024-05-03 11:15:00 | 1438.55 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-04-26 09:15:00 | 1462.28 | 2024-05-03 11:15:00 | 1438.55 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-04-26 11:15:00 | 1461.15 | 2024-05-03 11:15:00 | 1438.55 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-04-29 09:15:00 | 1464.53 | 2024-05-03 11:15:00 | 1438.55 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-05-21 12:45:00 | 1440.83 | 2024-05-30 13:15:00 | 1424.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-21 13:45:00 | 1437.70 | 2024-05-30 13:15:00 | 1424.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-05-22 09:15:00 | 1460.65 | 2024-05-30 13:15:00 | 1424.50 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-05-31 09:15:00 | 1436.45 | 2024-06-04 10:15:00 | 1403.15 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-06-07 11:30:00 | 1450.08 | 2024-07-05 14:15:00 | 1591.98 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2024-06-21 14:45:00 | 1447.98 | 2024-07-05 15:15:00 | 1595.09 | TARGET_HIT | 1.00 | 10.16% |
| BUY | retest2 | 2024-06-24 11:00:00 | 1447.25 | 2024-07-05 15:15:00 | 1592.78 | TARGET_HIT | 1.00 | 10.06% |
| BUY | retest2 | 2024-06-24 12:15:00 | 1447.85 | 2024-07-05 15:15:00 | 1592.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 09:15:00 | 1457.33 | 2024-07-08 11:15:00 | 1603.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 09:15:00 | 1465.00 | 2024-09-06 09:15:00 | 1468.50 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-08-06 14:45:00 | 1456.48 | 2024-09-06 09:15:00 | 1468.50 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2024-08-06 15:15:00 | 1457.78 | 2024-09-06 09:15:00 | 1468.50 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-08-20 09:15:00 | 1495.30 | 2024-09-06 09:15:00 | 1468.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1496.25 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-08-22 10:30:00 | 1494.10 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-08-28 10:45:00 | 1497.50 | 2024-09-11 12:15:00 | 1464.88 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-10-01 10:15:00 | 1470.78 | 2024-10-04 09:15:00 | 1397.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:15:00 | 1470.78 | 2024-10-24 10:15:00 | 1323.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-07-16 12:00:00 | 1482.80 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-07-16 14:00:00 | 1485.00 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest1 | 2025-07-17 10:45:00 | 1481.90 | 2025-07-21 11:15:00 | 1437.90 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1412.50 | 2025-08-20 10:15:00 | 1422.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-08-20 13:15:00 | 1419.10 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-20 14:15:00 | 1417.20 | 2025-08-21 09:15:00 | 1429.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-01 09:15:00 | 1345.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:15:00 | 1412.60 | 2025-09-01 09:15:00 | 1341.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 1416.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2025-08-22 12:15:00 | 1412.60 | 2025-09-12 10:15:00 | 1394.00 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-08-22 14:30:00 | 1410.50 | 2025-09-17 09:15:00 | 1410.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1410.10 | 2025-10-17 09:15:00 | 1407.80 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-08-25 11:00:00 | 1411.10 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-16 13:45:00 | 1399.90 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-22 14:15:00 | 1394.60 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-02-27 09:15:00 | 1388.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-02-27 09:15:00 | 1389.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-02-27 09:15:00 | 1389.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 1462.00 | 2026-03-04 09:15:00 | 1315.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 1462.60 | 2026-03-04 09:15:00 | 1316.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 1462.20 | 2026-03-04 09:15:00 | 1315.98 | TARGET_HIT | 0.50 | 10.00% |
