# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1671.00
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
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 15
- **Target hits / Stop hits / Partials:** 5 / 23 / 9
- **Avg / median % per leg:** 1.59% / 2.12%
- **Sum % (uncompounded):** 58.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.41% | 16.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.41% | 16.9% |
| SELL (all) | 25 | 18 | 72.0% | 1 | 15 | 9 | 1.68% | 42.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 18 | 72.0% | 1 | 15 | 9 | 1.68% | 42.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 22 | 59.5% | 5 | 23 | 9 | 1.59% | 58.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1270.10 | 1465.71 | 1465.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 1264.80 | 1429.68 | 1446.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 14:15:00 | 1292.00 | 1291.99 | 1345.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 1292.00 | 1291.99 | 1345.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1339.85 | 1293.36 | 1338.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1341.65 | 1293.36 | 1338.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1342.90 | 1293.86 | 1338.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1314.95 | 1300.33 | 1339.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:15:00 | 1249.20 | 1298.75 | 1335.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 14:15:00 | 1183.46 | 1282.67 | 1322.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 1375.00 | 1290.31 | 1290.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 1397.35 | 1299.12 | 1294.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 1448.55 | 1455.98 | 1407.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 11:00:00 | 1448.55 | 1455.98 | 1407.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1378.50 | 1454.22 | 1409.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 12:30:00 | 1447.60 | 1449.43 | 1409.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 1470.75 | 1448.19 | 1410.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-16 14:15:00 | 1592.36 | 1462.65 | 1421.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 1490.50 | 1549.86 | 1550.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 11:15:00 | 1476.80 | 1546.04 | 1548.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1527.40 | 1499.79 | 1519.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1523.50 | 1500.03 | 1519.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 1520.00 | 1500.69 | 1519.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 1520.00 | 1500.90 | 1519.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 1519.80 | 1501.62 | 1519.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1512.30 | 1502.48 | 1519.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1527.90 | 1503.00 | 1519.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 1521.30 | 1504.01 | 1519.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:30:00 | 1524.90 | 1505.73 | 1519.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1522.30 | 1506.20 | 1519.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1524.60 | 1506.78 | 1519.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1444.00 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1444.00 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1443.81 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1445.23 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1448.65 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1446.18 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1448.37 | 1504.52 | 1517.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 1436.68 | 1501.74 | 1516.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 14:15:00 | 1487.70 | 1483.29 | 1503.16 | SL hit (close>ema200) qty=0.50 sl=1483.29 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1622.60 | 1515.28 | 1514.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1633.10 | 1517.52 | 1516.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1681.40 | 1682.10 | 1636.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 1681.40 | 1682.10 | 1636.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1657.10 | 1692.39 | 1658.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 1654.20 | 1692.39 | 1658.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1666.50 | 1692.14 | 1658.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:00:00 | 1670.20 | 1691.62 | 1658.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 1672.00 | 1704.54 | 1675.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 1621.70 | 1701.39 | 1675.37 | SL hit (close<static) qty=1.00 sl=1656.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1611.30 | 1682.53 | 1682.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1569.70 | 1680.14 | 1681.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1719.30 | 1682.30 | 1682.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1742.80 | 1682.90 | 1682.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:45:00 | 1688.00 | 1697.82 | 1690.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1685.70 | 1697.70 | 1690.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:00:00 | 1697.50 | 1697.70 | 1690.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 1674.60 | 1700.40 | 1693.09 | SL hit (close<static) qty=1.00 sl=1684.10 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 1546.00 | 1689.08 | 1689.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1512.00 | 1672.92 | 1681.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1527.00 | 1506.87 | 1575.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:15:00 | 1534.50 | 1506.87 | 1575.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1569.10 | 1511.50 | 1573.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1536.50 | 1515.39 | 1573.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 1544.00 | 1516.71 | 1570.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 1597.30 | 1518.53 | 1570.77 | SL hit (close>static) qty=1.00 sl=1578.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-31 09:15:00 | 1229.05 | 2024-06-10 09:15:00 | 1351.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 1234.20 | 2024-06-10 09:15:00 | 1357.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1314.95 | 2024-12-17 10:15:00 | 1249.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1314.95 | 2024-12-20 14:15:00 | 1183.46 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-08 12:30:00 | 1447.60 | 2025-04-16 14:15:00 | 1592.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 09:15:00 | 1470.75 | 2025-04-16 14:15:00 | 1617.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-18 13:30:00 | 1520.00 | 2025-08-28 09:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 15:15:00 | 1520.00 | 2025-08-28 09:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-19 12:00:00 | 1519.80 | 2025-08-28 09:15:00 | 1443.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1512.30 | 2025-08-28 09:15:00 | 1445.23 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1521.30 | 2025-08-28 09:15:00 | 1448.65 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2025-08-22 11:30:00 | 1524.90 | 2025-08-28 09:15:00 | 1446.18 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1522.30 | 2025-08-28 09:15:00 | 1448.37 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1524.60 | 2025-08-28 14:15:00 | 1436.68 | PARTIAL | 0.50 | 5.77% |
| SELL | retest2 | 2025-08-18 13:30:00 | 1520.00 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-08-18 15:15:00 | 1520.00 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.12% |
| SELL | retest2 | 2025-08-19 12:00:00 | 1519.80 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1512.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1521.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2025-08-22 11:30:00 | 1524.90 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1522.30 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1524.60 | 2025-09-05 14:15:00 | 1487.70 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2025-09-08 14:45:00 | 1496.20 | 2025-09-09 09:15:00 | 1509.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-11 15:00:00 | 1494.30 | 2025-09-12 09:15:00 | 1513.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-04 13:00:00 | 1670.20 | 2025-12-19 12:15:00 | 1621.70 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-12-18 10:15:00 | 1672.00 | 2025-12-19 12:15:00 | 1621.70 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-12-23 09:15:00 | 1674.10 | 2026-01-20 13:15:00 | 1650.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-02-16 14:00:00 | 1697.50 | 2026-02-19 14:15:00 | 1674.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-23 09:45:00 | 1700.70 | 2026-02-24 09:15:00 | 1673.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-02-24 10:15:00 | 1696.10 | 2026-03-04 09:15:00 | 1640.80 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-02-24 10:45:00 | 1697.00 | 2026-03-04 09:15:00 | 1640.80 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1730.80 | 2026-03-06 14:15:00 | 1623.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1536.50 | 2026-04-16 09:15:00 | 1597.30 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-04-15 13:15:00 | 1544.00 | 2026-04-16 09:15:00 | 1597.30 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1543.00 | 2026-05-04 09:15:00 | 1634.70 | STOP_HIT | 1.00 | -5.94% |
| SELL | retest2 | 2026-04-23 11:30:00 | 1540.80 | 2026-05-04 09:15:00 | 1634.70 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1502.90 | 2026-05-04 09:15:00 | 1634.70 | STOP_HIT | 1.00 | -8.77% |
