# TECHM (TECHM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1450.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 38 |
| PENDING_CANCEL | 19 |
| ENTRY1 | 3 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 0 / 18 / 1
- **Avg / median % per leg:** 0.58% / -2.46%
- **Sum % (uncompounded):** 10.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 6 | 31.6% | 0 | 18 | 1 | 0.58% | 11.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.53% | -10.6% |
| SELL @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 0 | 15 | 1 | 1.35% | 21.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.53% | -10.6% |
| retest2 (combined) | 16 | 6 | 37.5% | 0 | 15 | 1 | 1.35% | 21.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 1652.05 | 1690.97 | 1691.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1633.20 | 1685.34 | 1688.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 1664.60 | 1679.70 | 1684.16 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-11 13:15:00 | 1675.45 | 1679.66 | 1684.12 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 14:15:00 | 1669.95 | 1679.57 | 1684.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-12 09:15:00 | 1682.85 | 1679.52 | 1683.98 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-02-14 10:15:00 | 1661.15 | 1679.48 | 1683.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:15:00 | 1660.45 | 1679.17 | 1683.44 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-02-18 12:15:00 | 1688.80 | 1677.21 | 1682.12 | SL hit (close>static) qty=1.00 sl=1688.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-20 11:15:00 | 1670.80 | 1679.10 | 1682.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 13:15:00 | 1659.75 | 1678.75 | 1682.59 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-12 12:15:00 | 1410.79 | 1584.27 | 1625.96 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1441.80 | 1397.49 | 1476.18 | SL hit (close>ema200) qty=0.50 sl=1397.49 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-19 09:15:00 | 1665.50 | 1596.28 | 1561.71 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-19 11:15:00 | 1687.80 | 1597.92 | 1562.87 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 1673.50 | 1602.05 | 1565.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-20 10:15:00 | 1682.30 | 1602.85 | 1566.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-30 09:15:00 | 1663.30 | 1632.03 | 1589.27 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-30 11:15:00 | 1674.20 | 1632.77 | 1590.07 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-07-01 11:15:00 | 1672.90 | 1635.98 | 1593.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1671.30 | 1636.69 | 1593.95 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-02 10:15:00 | 1670.50 | 1638.14 | 1595.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-02 11:15:00 | 1677.60 | 1638.53 | 1595.94 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1689.00 | 1640.54 | 1598.01 | SL hit (close>static) qty=1.00 sl=1688.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-03 14:15:00 | 1670.50 | 1642.34 | 1599.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-03 15:15:00 | 1675.00 | 1642.66 | 1600.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-04 09:15:00 | 1654.80 | 1642.78 | 1600.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 1658.30 | 1643.10 | 1601.19 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1657.90 | 1643.25 | 1601.48 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-04 13:15:00 | 1651.50 | 1643.33 | 1601.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-04 14:15:00 | 1656.50 | 1643.46 | 1602.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-04 15:15:00 | 1655.00 | 1643.58 | 1602.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 1635.50 | 1643.50 | 1602.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1460.30 | 1585.43 | 1585.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1460.30 | 1585.43 | 1585.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1460.30 | 1585.43 | 1585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1447.50 | 1581.51 | 1583.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.51 | 1544.80 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-14 11:15:00 | 1502.60 | 1520.33 | 1544.59 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 13:15:00 | 1491.80 | 1519.80 | 1544.08 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-22 10:15:00 | 1499.80 | 1515.15 | 1537.87 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-22 11:15:00 | 1507.70 | 1515.08 | 1537.72 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-22 15:15:00 | 1503.00 | 1514.80 | 1537.13 | ENTRY1 cross detected — sustain check pending (75m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.93 | 1537.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-26 14:15:00 | 1500.90 | 1515.70 | 1536.20 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1484.30 | 1515.28 | 1535.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 2580m) |
| Cross detected — sustain check pending | 2025-09-03 10:15:00 | 1503.00 | 1512.10 | 1531.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-03 11:15:00 | 1508.00 | 1512.06 | 1531.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 1501.90 | 1511.95 | 1531.02 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-03 13:15:00 | 1505.90 | 1511.89 | 1530.89 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 1499.50 | 1511.71 | 1530.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1503.60 | 1511.51 | 1530.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-04 13:15:00 | 1502.20 | 1511.36 | 1529.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1496.90 | 1511.10 | 1529.65 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1528.50 | 1505.38 | 1524.67 | SL hit (close>ema400) qty=1.00 sl=1524.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1541.80 | 1511.03 | 1524.40 | SL hit (close>static) qty=1.00 sl=1539.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1541.80 | 1511.03 | 1524.40 | SL hit (close>static) qty=1.00 sl=1539.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1541.80 | 1511.03 | 1524.40 | SL hit (close>static) qty=1.00 sl=1539.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 1498.10 | 1516.85 | 1526.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 1503.40 | 1516.51 | 1526.04 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1484.30 | 1464.17 | 1485.62 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-23 14:15:00 | 1462.00 | 1464.64 | 1485.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-23 15:15:00 | 1465.30 | 1464.65 | 1485.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-24 10:15:00 | 1461.70 | 1464.61 | 1485.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 1460.60 | 1464.54 | 1484.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-27 14:15:00 | 1462.40 | 1464.37 | 1483.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-27 15:15:00 | 1464.00 | 1464.37 | 1483.69 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-28 09:15:00 | 1455.60 | 1464.28 | 1483.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 1443.30 | 1463.92 | 1483.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-11-21 12:15:00 | 1455.90 | 1442.41 | 1459.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-21 13:15:00 | 1465.80 | 1442.65 | 1459.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-21 14:15:00 | 1462.00 | 1442.84 | 1460.00 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-24 09:15:00 | 1510.50 | 1443.70 | 1460.26 | ENTRY2 sustain failed after 4020m |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1510.50 | 1443.70 | 1460.26 | SL hit (close>static) qty=1.00 sl=1487.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1510.50 | 1443.70 | 1460.26 | SL hit (close>static) qty=1.00 sl=1487.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 1546.30 | 1472.19 | 1472.43 | SL hit (close>static) qty=1.00 sl=1539.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1460.90 | 1605.39 | 1597.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-20 10:15:00 | 1470.60 | 1604.04 | 1596.89 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-20 12:15:00 | 1462.90 | 1601.32 | 1595.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:15:00 | 1455.80 | 1598.43 | 1594.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 1442.30 | 1595.56 | 1592.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1450.00 | 1592.57 | 1591.32 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 1439.60 | 1588.09 | 1589.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.16 | 1473.32 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-25 15:15:00 | 1403.50 | 1416.80 | 1470.60 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1407.50 | 1416.71 | 1470.28 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 2520m) |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1406.30 | 1413.21 | 1463.83 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 15:15:00 | 1405.70 | 1413.05 | 1463.25 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.81 | 1461.66 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1463.80 | 1416.51 | 1461.38 | SL hit (close>ema400) qty=1.00 sl=1461.38 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1463.80 | 1416.51 | 1461.38 | SL hit (close>ema400) qty=1.00 sl=1461.38 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-10 10:15:00 | 1431.00 | 1424.46 | 1460.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 1437.20 | 1424.65 | 1460.67 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1423.30 | 1425.02 | 1460.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1433.70 | 1425.10 | 1460.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1422.60 | 1446.59 | 1465.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1414.90 | 1445.92 | 1464.60 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1471.80 | 1446.17 | 1464.63 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1425.00 | 1446.29 | 1464.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-23 10:15:00 | 1434.60 | 1446.17 | 1464.27 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 13:15:00 | 1428.70 | 1445.85 | 1463.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 1416.90 | 1445.35 | 1463.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 1466.70 | 1437.04 | 1456.01 | SL hit (close>static) qty=1.00 sl=1464.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-14 12:15:00 | 1660.45 | 2025-02-18 12:15:00 | 1688.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-02-20 13:15:00 | 1659.75 | 2025-03-12 12:15:00 | 1410.79 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-20 13:15:00 | 1659.75 | 2025-04-23 09:15:00 | 1441.80 | STOP_HIT | 0.50 | 13.13% |
| SELL | retest2 | 2025-07-01 13:15:00 | 1671.30 | 2025-07-03 09:15:00 | 1689.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-04 11:15:00 | 1658.30 | 2025-07-28 10:15:00 | 1460.30 | STOP_HIT | 1.00 | 11.94% |
| SELL | retest2 | 2025-07-07 09:15:00 | 1635.50 | 2025-07-28 10:15:00 | 1460.30 | STOP_HIT | 1.00 | 10.71% |
| SELL | retest1 | 2025-08-14 13:15:00 | 1491.80 | 2025-09-10 09:15:00 | 1528.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1484.30 | 2025-09-17 12:15:00 | 1541.80 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1503.60 | 2025-09-17 12:15:00 | 1541.80 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-09-04 15:15:00 | 1496.90 | 2025-09-17 12:15:00 | 1541.80 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-22 11:15:00 | 1503.40 | 2025-11-24 09:15:00 | 1510.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-10-24 12:15:00 | 1460.60 | 2025-11-24 09:15:00 | 1510.50 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-10-28 11:15:00 | 1443.30 | 2025-12-03 10:15:00 | 1546.30 | STOP_HIT | 1.00 | -7.14% |
| SELL | retest2 | 2026-02-20 14:15:00 | 1455.80 | 2026-02-23 13:15:00 | 1441.90 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1450.00 | 2026-02-23 13:15:00 | 1441.90 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest1 | 2026-03-27 09:15:00 | 1407.50 | 2026-04-07 09:15:00 | 1463.80 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest1 | 2026-04-01 15:15:00 | 1405.70 | 2026-04-07 09:15:00 | 1463.80 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2026-04-22 12:15:00 | 1414.90 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-30 12:15:00 | 1466.70 | STOP_HIT | 1.00 | -3.51% |
