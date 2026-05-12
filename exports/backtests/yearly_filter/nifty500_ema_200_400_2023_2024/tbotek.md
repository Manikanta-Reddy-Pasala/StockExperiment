# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2024-05-15 09:15:00 → 2026-05-11 15:15:00 (3437 bars)
- **Last close:** 1195.10
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
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 13
- **Target hits / Stop hits / Partials:** 6 / 19 / 11
- **Avg / median % per leg:** 2.50% / 4.32%
- **Sum % (uncompounded):** 89.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | -2.15% | -15.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -2.15% | -15.1% |
| SELL (all) | 29 | 22 | 75.9% | 5 | 13 | 11 | 3.62% | 104.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 22 | 75.9% | 5 | 13 | 11 | 3.62% | 104.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 23 | 63.9% | 6 | 19 | 11 | 2.50% | 89.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 1612.40 | 1712.14 | 1712.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1575.45 | 1705.40 | 1708.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 1679.85 | 1663.32 | 1684.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1679.85 | 1663.32 | 1684.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1679.85 | 1663.32 | 1684.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 1700.05 | 1663.32 | 1684.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1705.10 | 1663.73 | 1684.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 1705.10 | 1663.73 | 1684.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1684.25 | 1663.94 | 1684.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:15:00 | 1653.00 | 1672.21 | 1686.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 14:15:00 | 1570.35 | 1663.49 | 1680.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-26 12:15:00 | 1487.70 | 1621.54 | 1654.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1750.00 | 1644.09 | 1643.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 1767.75 | 1654.84 | 1649.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 1639.85 | 1700.16 | 1676.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 1639.85 | 1700.16 | 1676.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1639.85 | 1700.16 | 1676.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 1639.85 | 1700.16 | 1676.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1647.90 | 1699.64 | 1676.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:30:00 | 1612.45 | 1699.64 | 1676.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1725.20 | 1697.49 | 1676.08 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1570.00 | 1663.86 | 1664.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1501.00 | 1662.24 | 1663.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 14:15:00 | 1640.70 | 1631.91 | 1645.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 14:15:00 | 1640.70 | 1631.91 | 1645.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1640.70 | 1631.91 | 1645.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:45:00 | 1648.80 | 1631.91 | 1645.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 1649.10 | 1632.08 | 1645.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 1651.15 | 1632.08 | 1645.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1671.15 | 1632.47 | 1645.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 1674.15 | 1632.47 | 1645.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1660.25 | 1632.74 | 1645.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:15:00 | 1640.70 | 1632.74 | 1645.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:15:00 | 1651.40 | 1632.64 | 1644.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:15:00 | 1568.83 | 1631.91 | 1643.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:15:00 | 1558.66 | 1631.17 | 1642.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 15:15:00 | 1628.00 | 1627.04 | 1640.04 | SL hit (close>ema200) qty=0.50 sl=1627.04 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 1363.90 | 1258.51 | 1258.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 14:15:00 | 1386.30 | 1277.55 | 1269.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1318.40 | 1331.59 | 1303.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:45:00 | 1320.00 | 1331.59 | 1303.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1340.90 | 1371.28 | 1339.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 1358.60 | 1371.28 | 1339.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 1360.00 | 1371.14 | 1339.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1366.10 | 1370.57 | 1340.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:00:00 | 1358.50 | 1374.20 | 1344.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1397.50 | 1373.35 | 1347.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1416.60 | 1374.99 | 1349.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 1413.30 | 1409.57 | 1376.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 1320.80 | 1401.55 | 1375.09 | SL hit (close<static) qty=1.00 sl=1330.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1478.30 | 1607.64 | 1607.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1465.20 | 1606.22 | 1607.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1495.00 | 1494.71 | 1537.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 1493.70 | 1494.71 | 1537.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1559.00 | 1496.64 | 1536.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1559.00 | 1496.64 | 1536.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1572.80 | 1497.40 | 1536.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1572.80 | 1497.40 | 1536.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1543.90 | 1504.23 | 1538.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 1543.90 | 1504.23 | 1538.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1539.00 | 1504.58 | 1538.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1448.20 | 1504.58 | 1538.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1482.80 | 1504.36 | 1538.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1424.00 | 1502.57 | 1536.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1444.50 | 1499.71 | 1533.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:15:00 | 1442.70 | 1498.03 | 1531.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 1434.00 | 1497.53 | 1531.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:15:00 | 1372.27 | 1484.71 | 1522.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 1370.57 | 1479.33 | 1518.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1352.80 | 1476.98 | 1517.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1362.30 | 1476.98 | 1517.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 1300.05 | 1459.71 | 1505.37 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 15:15:00 | 1653.00 | 2024-11-14 14:15:00 | 1570.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 15:15:00 | 1653.00 | 2024-11-26 12:15:00 | 1487.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 1669.45 | 2024-12-17 10:15:00 | 1737.20 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2024-12-16 11:30:00 | 1673.80 | 2024-12-17 10:15:00 | 1737.20 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-12-20 15:00:00 | 1672.55 | 2024-12-24 13:15:00 | 1728.30 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-02-06 11:15:00 | 1640.70 | 2025-02-12 10:15:00 | 1568.83 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1651.40 | 2025-02-13 09:15:00 | 1558.66 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2025-02-06 11:15:00 | 1640.70 | 2025-02-14 15:15:00 | 1628.00 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1651.40 | 2025-02-14 15:15:00 | 1628.00 | STOP_HIT | 0.50 | 1.42% |
| BUY | retest2 | 2025-07-31 10:15:00 | 1358.60 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-31 10:45:00 | 1360.00 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1366.10 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-08-05 12:00:00 | 1358.50 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1416.60 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2025-08-25 14:45:00 | 1413.30 | 2025-08-29 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -6.54% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1550.00 | 2025-11-20 10:15:00 | 1705.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1424.00 | 2026-02-19 09:15:00 | 1372.27 | PARTIAL | 0.50 | 3.63% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1444.50 | 2026-02-19 14:15:00 | 1370.57 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-02-16 14:15:00 | 1442.70 | 2026-02-20 09:15:00 | 1352.80 | PARTIAL | 0.50 | 6.23% |
| SELL | retest2 | 2026-02-16 15:15:00 | 1434.00 | 2026-02-20 09:15:00 | 1362.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1424.00 | 2026-02-24 09:15:00 | 1300.05 | TARGET_HIT | 0.50 | 8.70% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1444.50 | 2026-02-25 09:15:00 | 1298.43 | TARGET_HIT | 0.50 | 10.11% |
| SELL | retest2 | 2026-02-16 14:15:00 | 1442.70 | 2026-02-25 10:15:00 | 1281.60 | TARGET_HIT | 0.50 | 11.17% |
| SELL | retest2 | 2026-02-16 15:15:00 | 1434.00 | 2026-02-25 10:15:00 | 1290.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 13:30:00 | 1290.10 | 2026-04-24 09:15:00 | 1225.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 13:30:00 | 1290.10 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2026-04-17 15:00:00 | 1291.80 | 2026-04-24 09:15:00 | 1227.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 15:00:00 | 1291.80 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2026-04-20 12:00:00 | 1289.70 | 2026-04-24 09:15:00 | 1225.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 12:00:00 | 1289.70 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1292.10 | 2026-04-24 09:15:00 | 1227.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1292.10 | 2026-04-24 09:15:00 | 1236.00 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2026-04-28 11:30:00 | 1258.50 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-04-30 10:00:00 | 1255.00 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-30 11:45:00 | 1257.00 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1259.30 | 2026-05-04 10:15:00 | 1282.80 | STOP_HIT | 1.00 | -1.87% |
