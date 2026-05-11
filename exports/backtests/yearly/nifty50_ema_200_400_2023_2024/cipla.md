# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2022-04-07 09:15:00 → 2026-05-08 15:15:00 (7054 bars)
- **Last close:** 1348.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 38
- **Target hits / Stop hits / Partials:** 2 / 48 / 8
- **Avg / median % per leg:** -0.13% / -1.35%
- **Sum % (uncompounded):** -7.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 4 | 13.3% | 2 | 28 | 0 | -0.87% | -26.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 4 | 13.3% | 2 | 28 | 0 | -0.87% | -26.1% |
| SELL (all) | 28 | 16 | 57.1% | 0 | 20 | 8 | 0.67% | 18.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 16 | 57.1% | 0 | 20 | 8 | 0.67% | 18.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 20 | 34.5% | 2 | 48 | 8 | -0.13% | -7.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 10:15:00 | 972.10 | 937.93 | 937.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 15:15:00 | 973.50 | 939.53 | 938.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 1217.50 | 1221.26 | 1170.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 10:00:00 | 1217.50 | 1221.26 | 1170.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1179.50 | 1220.44 | 1172.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 1179.50 | 1220.44 | 1172.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1180.50 | 1220.04 | 1172.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:45:00 | 1176.00 | 1220.04 | 1172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 1175.55 | 1214.40 | 1172.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:45:00 | 1171.50 | 1214.40 | 1172.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 1171.10 | 1213.97 | 1172.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 14:00:00 | 1171.10 | 1213.97 | 1172.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 1164.05 | 1213.47 | 1172.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 1164.05 | 1213.47 | 1172.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 1170.00 | 1209.99 | 1172.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 10:45:00 | 1171.00 | 1209.99 | 1172.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 1165.45 | 1209.55 | 1172.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:00:00 | 1165.45 | 1209.55 | 1172.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 1164.30 | 1207.78 | 1172.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:30:00 | 1174.85 | 1207.49 | 1172.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 12:15:00 | 1159.90 | 1200.86 | 1173.25 | SL hit (close<static) qty=1.00 sl=1160.05 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 15:15:00 | 1338.05 | 1408.38 | 1408.50 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1419.30 | 1408.69 | 1408.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 10:15:00 | 1371.65 | 1408.34 | 1408.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 14:15:00 | 1359.25 | 1406.81 | 1407.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 1419.65 | 1406.46 | 1407.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 1419.65 | 1406.46 | 1407.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1419.65 | 1406.46 | 1407.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:15:00 | 1407.90 | 1406.46 | 1407.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1417.55 | 1406.57 | 1407.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 1400.55 | 1406.62 | 1407.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 1402.55 | 1406.52 | 1407.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 1400.20 | 1406.78 | 1407.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-18 09:45:00 | 1403.00 | 1406.46 | 1407.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1416.65 | 1406.50 | 1407.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1416.65 | 1406.50 | 1407.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1418.55 | 1406.62 | 1407.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-21 13:15:00 | 1436.70 | 1407.24 | 1407.78 | SL hit (close>static) qty=1.00 sl=1428.45 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 1459.00 | 1408.42 | 1408.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 10:15:00 | 1465.65 | 1408.99 | 1408.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1437.80 | 1439.20 | 1426.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 12:00:00 | 1437.80 | 1439.20 | 1426.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1448.30 | 1439.29 | 1426.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 1465.20 | 1439.72 | 1427.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-27 09:15:00 | 1611.72 | 1552.81 | 1526.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1480.20 | 1586.62 | 1586.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1457.85 | 1574.08 | 1580.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1559.80 | 1557.80 | 1571.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:30:00 | 1556.95 | 1557.80 | 1571.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1569.05 | 1557.89 | 1571.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1569.05 | 1557.89 | 1571.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1576.80 | 1558.09 | 1571.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:45:00 | 1562.15 | 1564.42 | 1573.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:30:00 | 1561.70 | 1566.31 | 1573.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:00:00 | 1562.15 | 1566.10 | 1573.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:30:00 | 1561.80 | 1565.98 | 1573.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1484.04 | 1555.42 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1483.62 | 1555.42 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1484.04 | 1555.42 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1483.71 | 1555.42 | 1566.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 1538.80 | 1525.80 | 1546.92 | SL hit (close>ema200) qty=0.50 sl=1525.80 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 13:15:00 | 1523.55 | 1471.63 | 1471.52 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 11:15:00 | 1443.90 | 1471.98 | 1472.00 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1499.95 | 1472.12 | 1472.07 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1410.00 | 1471.58 | 1471.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1467.95 | 1469.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:45:00 | 1455.35 | 1457.08 | 1463.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 1494.60 | 1458.68 | 1464.33 | SL hit (close>static) qty=1.00 sl=1488.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.82 | 1469.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1475.92 | 1472.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.94 | 1488.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-07 09:45:00 | 1502.60 | 1501.94 | 1488.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1490.00 | 1502.27 | 1489.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:15:00 | 1491.20 | 1502.27 | 1489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1488.40 | 1502.13 | 1489.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:30:00 | 1484.10 | 1502.13 | 1489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 1481.90 | 1501.93 | 1489.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 1481.90 | 1501.93 | 1489.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1484.50 | 1501.76 | 1489.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:30:00 | 1480.10 | 1501.76 | 1489.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1487.20 | 1501.46 | 1489.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 1486.00 | 1501.46 | 1489.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1480.00 | 1501.25 | 1488.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1465.50 | 1501.25 | 1488.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1503.60 | 1501.44 | 1490.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 1506.00 | 1500.39 | 1490.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 1510.70 | 1500.35 | 1490.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 1471.40 | 1499.64 | 1490.80 | SL hit (close<static) qty=1.00 sl=1474.60 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1475.30 | 1493.10 | 1493.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1472.20 | 1492.23 | 1492.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1541.60 | 1488.39 | 1490.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1532.00 | 1488.82 | 1490.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1532.10 | 1488.82 | 1490.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.20 | 1492.84 | 1492.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.30 | 1493.64 | 1493.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1506.70 | 1507.04 | 1500.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 12:00:00 | 1506.70 | 1507.04 | 1500.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1504.00 | 1506.97 | 1500.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:15:00 | 1500.10 | 1506.97 | 1500.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1495.20 | 1506.86 | 1500.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1495.20 | 1506.86 | 1500.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1505.00 | 1506.84 | 1500.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1511.60 | 1506.84 | 1500.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1481.30 | 1506.64 | 1500.74 | SL hit (close<static) qty=1.00 sl=1494.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.14 | 1531.16 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.34 | 1531.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1568.00 | 1531.98 | 1531.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1524.90 | 1539.68 | 1539.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.12 | 1539.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.00 | 1529.74 | 1533.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 09:15:00 | 901.00 | 2023-05-24 12:15:00 | 935.35 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2023-09-29 09:30:00 | 1174.85 | 2023-10-05 12:15:00 | 1159.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-10-06 09:15:00 | 1173.00 | 2023-10-09 10:15:00 | 1158.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-10-11 13:30:00 | 1172.50 | 2023-10-12 11:15:00 | 1157.25 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-10-11 14:00:00 | 1173.05 | 2023-10-12 11:15:00 | 1157.25 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-10-13 10:00:00 | 1167.00 | 2023-10-25 09:15:00 | 1166.40 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2023-10-13 12:15:00 | 1165.60 | 2023-10-25 09:15:00 | 1166.40 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2023-10-13 14:00:00 | 1165.20 | 2023-10-25 09:15:00 | 1166.40 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2023-10-16 09:45:00 | 1167.70 | 2023-10-25 09:15:00 | 1166.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-10-17 11:30:00 | 1176.05 | 2023-10-25 12:15:00 | 1156.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-10-17 12:30:00 | 1174.50 | 2023-10-25 12:15:00 | 1156.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-10-17 14:15:00 | 1174.65 | 2023-10-25 12:15:00 | 1156.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-10-18 09:15:00 | 1195.55 | 2023-10-25 12:15:00 | 1156.20 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2023-10-27 13:00:00 | 1170.55 | 2024-01-03 09:15:00 | 1287.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-16 09:15:00 | 1400.55 | 2024-05-21 13:15:00 | 1436.70 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-05-16 14:00:00 | 1402.55 | 2024-05-21 13:15:00 | 1436.70 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-05-17 12:15:00 | 1400.20 | 2024-05-21 13:15:00 | 1436.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-05-18 09:45:00 | 1403.00 | 2024-05-21 13:15:00 | 1436.70 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-06-04 15:00:00 | 1465.20 | 2024-08-27 09:15:00 | 1611.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-07 09:45:00 | 1562.15 | 2024-11-18 09:15:00 | 1484.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:30:00 | 1561.70 | 2024-11-18 09:15:00 | 1483.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 14:00:00 | 1562.15 | 2024-11-18 09:15:00 | 1484.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 14:30:00 | 1561.80 | 2024-11-18 09:15:00 | 1483.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:45:00 | 1562.15 | 2024-11-29 10:15:00 | 1538.80 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2024-11-11 11:30:00 | 1561.70 | 2024-11-29 10:15:00 | 1538.80 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2024-11-11 14:00:00 | 1562.15 | 2024-11-29 10:15:00 | 1538.80 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2024-11-11 14:30:00 | 1561.80 | 2024-11-29 10:15:00 | 1538.80 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2024-12-02 13:00:00 | 1510.30 | 2024-12-13 09:15:00 | 1434.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 10:30:00 | 1514.00 | 2024-12-13 09:15:00 | 1438.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 13:00:00 | 1510.30 | 2024-12-19 11:15:00 | 1493.00 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2024-12-04 10:30:00 | 1514.00 | 2024-12-19 11:15:00 | 1493.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-01-03 11:30:00 | 1512.95 | 2025-01-13 14:15:00 | 1437.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 14:15:00 | 1513.45 | 2025-01-13 14:15:00 | 1437.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 11:30:00 | 1512.95 | 2025-01-28 13:15:00 | 1463.25 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-01-03 14:15:00 | 1513.45 | 2025-01-28 13:15:00 | 1463.25 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-02-01 12:15:00 | 1447.75 | 2025-02-13 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-02-04 14:15:00 | 1451.50 | 2025-02-13 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-02-05 10:00:00 | 1446.30 | 2025-02-13 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1437.75 | 2025-02-13 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1422.80 | 2025-03-10 09:15:00 | 1478.60 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-02-28 13:15:00 | 1427.90 | 2025-03-10 09:15:00 | 1478.60 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-04-11 10:45:00 | 1455.35 | 2025-04-16 09:15:00 | 1494.60 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-05-15 14:00:00 | 1506.00 | 2025-05-20 09:15:00 | 1471.40 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-05-19 09:45:00 | 1510.70 | 2025-05-20 09:15:00 | 1471.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-06-09 12:45:00 | 1506.60 | 2025-07-08 09:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-06-09 15:00:00 | 1506.60 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-16 09:15:00 | 1526.70 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-06-17 13:45:00 | 1510.40 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-18 09:30:00 | 1512.80 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-06-24 09:45:00 | 1511.30 | 2025-07-09 10:15:00 | 1491.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1509.40 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-06-30 13:00:00 | 1507.20 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-07-01 13:00:00 | 1507.90 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-07-02 11:00:00 | 1507.00 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-07-03 09:15:00 | 1500.50 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-07-09 09:30:00 | 1498.90 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1511.60 | 2025-08-06 09:15:00 | 1481.30 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1508.10 | 2025-09-26 10:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.99% |
