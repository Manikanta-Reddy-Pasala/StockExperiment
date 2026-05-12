# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1261.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 1 |
| TARGET_HIT | 8 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 8 / 19 / 1
- **Avg / median % per leg:** 2.11% / -0.79%
- **Sum % (uncompounded):** 59.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 8 | 10 | 0 | 3.76% | 67.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 8 | 10 | 0 | 3.76% | 67.6% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.85% | -8.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.85% | -8.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 10 | 35.7% | 8 | 19 | 1 | 2.11% | 59.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 15:15:00 | 535.00 | 501.03 | 500.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 11:15:00 | 535.73 | 514.46 | 509.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 09:15:00 | 591.90 | 594.86 | 572.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-14 09:30:00 | 590.05 | 594.86 | 572.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 573.45 | 593.02 | 573.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 573.45 | 593.02 | 573.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 576.88 | 592.86 | 573.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 11:15:00 | 577.63 | 592.86 | 573.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:30:00 | 579.48 | 591.75 | 573.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 13:15:00 | 578.73 | 591.24 | 573.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 09:15:00 | 582.92 | 590.82 | 573.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 571.92 | 588.99 | 574.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 571.92 | 588.99 | 574.57 | SL hit (close<static) qty=1.00 sl=572.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 13:15:00 | 828.50 | 904.30 | 904.60 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 11:15:00 | 935.28 | 903.44 | 903.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 14:15:00 | 937.53 | 904.38 | 903.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-10 11:15:00 | 1003.60 | 1005.77 | 969.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-10 12:00:00 | 1003.60 | 1005.77 | 969.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 976.50 | 1005.18 | 970.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 974.00 | 1005.18 | 970.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1047.00 | 1037.35 | 1004.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1035.78 | 1037.35 | 1004.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 992.03 | 1036.90 | 1004.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 992.03 | 1036.90 | 1004.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 952.30 | 1036.06 | 1004.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 952.30 | 1036.06 | 1004.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1004.48 | 1035.74 | 1004.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 1011.00 | 1035.74 | 1004.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 1018.10 | 1030.94 | 1003.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 1010.40 | 1034.11 | 1014.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 11:45:00 | 1009.90 | 1029.81 | 1014.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-28 10:15:00 | 1112.10 | 1030.35 | 1015.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1364.00 | 1646.00 | 1646.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 1356.00 | 1640.33 | 1643.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1196.00 | 1195.23 | 1305.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 1196.00 | 1195.23 | 1305.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1281.00 | 1200.76 | 1259.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 1272.20 | 1200.76 | 1259.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1289.50 | 1201.65 | 1259.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:00:00 | 1289.50 | 1201.65 | 1259.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1281.40 | 1261.62 | 1278.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1281.40 | 1261.62 | 1278.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1291.50 | 1261.91 | 1278.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1291.50 | 1261.91 | 1278.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1277.50 | 1264.39 | 1279.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:45:00 | 1273.90 | 1264.43 | 1279.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:30:00 | 1264.10 | 1264.20 | 1279.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 1284.00 | 1264.85 | 1278.83 | SL hit (close>static) qty=1.00 sl=1282.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1456.00 | 1288.07 | 1287.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1472.30 | 1299.46 | 1293.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 1691.50 | 1700.87 | 1600.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1691.50 | 1700.87 | 1600.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1638.00 | 1700.85 | 1629.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 1637.00 | 1700.85 | 1629.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1630.00 | 1700.15 | 1629.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 1630.00 | 1700.15 | 1629.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1634.80 | 1699.50 | 1629.08 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1512.00 | 1590.57 | 1590.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1506.70 | 1589.73 | 1590.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1551.60 | 1551.09 | 1567.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:00:00 | 1551.60 | 1551.09 | 1567.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1562.30 | 1551.48 | 1566.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1562.40 | 1551.48 | 1566.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1568.00 | 1550.42 | 1563.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 1568.00 | 1550.42 | 1563.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1570.90 | 1550.63 | 1563.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 1574.70 | 1550.63 | 1563.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1569.20 | 1551.14 | 1563.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1569.00 | 1551.14 | 1563.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1569.60 | 1551.32 | 1563.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1579.50 | 1551.32 | 1563.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1567.70 | 1556.96 | 1565.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1570.50 | 1556.96 | 1565.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1572.70 | 1557.23 | 1565.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 1572.70 | 1557.23 | 1565.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1573.00 | 1527.98 | 1546.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 1588.40 | 1527.98 | 1546.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1577.70 | 1528.48 | 1546.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 1583.50 | 1528.48 | 1546.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1543.30 | 1531.13 | 1547.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 1547.70 | 1531.13 | 1547.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1535.20 | 1531.31 | 1547.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1532.60 | 1531.31 | 1547.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1557.40 | 1531.68 | 1547.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1557.40 | 1531.68 | 1547.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1559.80 | 1531.96 | 1547.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 1552.20 | 1532.16 | 1547.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1574.80 | 1533.08 | 1547.54 | SL hit (close>static) qty=1.00 sl=1564.80 alert=retest2 |

### Cycle 7 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1613.20 | 1559.64 | 1559.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1622.00 | 1570.48 | 1565.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1581.10 | 1581.94 | 1572.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1546.00 | 1583.17 | 1573.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1546.00 | 1583.17 | 1573.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1557.20 | 1582.92 | 1573.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 1566.80 | 1576.30 | 1570.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1559.90 | 1602.25 | 1590.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1537.10 | 1598.39 | 1589.08 | SL hit (close<static) qty=1.00 sl=1542.20 alert=retest2 |

### Cycle 8 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1491.80 | 1580.64 | 1580.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1482.50 | 1577.86 | 1579.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 1399.70 | 1379.94 | 1434.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:30:00 | 1411.60 | 1379.94 | 1434.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1284.50 | 1223.91 | 1284.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 1284.50 | 1223.91 | 1284.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1287.20 | 1224.54 | 1284.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 1287.20 | 1224.54 | 1284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 1282.80 | 1225.12 | 1284.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 1277.90 | 1225.65 | 1284.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 15:15:00 | 1291.00 | 1226.90 | 1284.65 | SL hit (close>static) qty=1.00 sl=1289.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-18 11:15:00 | 577.63 | 2023-08-25 10:15:00 | 571.92 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-08-21 09:30:00 | 579.48 | 2023-08-25 10:15:00 | 571.92 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-08-21 13:15:00 | 578.73 | 2023-08-25 10:15:00 | 571.92 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-08-22 09:15:00 | 582.92 | 2023-08-25 10:15:00 | 571.92 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2023-08-30 10:15:00 | 575.23 | 2023-08-30 13:15:00 | 571.25 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-08-31 09:15:00 | 578.00 | 2023-08-31 14:15:00 | 569.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-08-31 13:15:00 | 573.90 | 2023-08-31 14:15:00 | 569.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-08-31 14:00:00 | 573.78 | 2023-08-31 14:15:00 | 569.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-09-04 09:15:00 | 578.42 | 2023-09-08 12:15:00 | 636.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1011.00 | 2024-06-28 10:15:00 | 1112.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1018.10 | 2024-06-28 10:15:00 | 1119.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 09:45:00 | 1010.40 | 2024-06-28 10:15:00 | 1111.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-27 11:45:00 | 1009.90 | 2024-06-28 10:15:00 | 1110.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-08 13:15:00 | 1357.00 | 2024-10-10 09:15:00 | 1492.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-08 13:45:00 | 1358.50 | 2024-10-10 09:15:00 | 1494.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-08 14:45:00 | 1356.55 | 2024-10-10 09:15:00 | 1492.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-06 13:45:00 | 1273.90 | 2025-05-08 11:15:00 | 1284.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-05-07 09:30:00 | 1264.10 | 2025-05-08 11:15:00 | 1284.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1274.20 | 2025-05-09 09:15:00 | 1210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1274.20 | 2025-05-12 09:15:00 | 1266.80 | STOP_HIT | 0.50 | 0.58% |
| SELL | retest2 | 2025-10-09 14:00:00 | 1552.20 | 2025-10-10 09:15:00 | 1574.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-07 13:30:00 | 1566.80 | 2025-12-04 11:15:00 | 1537.10 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-03 10:45:00 | 1559.90 | 2025-12-04 11:15:00 | 1537.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-08 14:00:00 | 1277.90 | 2026-04-08 15:15:00 | 1291.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1269.10 | 2026-04-09 11:15:00 | 1290.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-04-09 13:30:00 | 1277.10 | 2026-04-10 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-04-09 14:00:00 | 1277.80 | 2026-04-10 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1274.90 | 2026-04-15 09:15:00 | 1333.70 | STOP_HIT | 1.00 | -4.61% |
