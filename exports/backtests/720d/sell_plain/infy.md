# INFY (INFY)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1165.90
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
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 24 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 2 / 12
- **Target hits / Stop hits / Partials:** 0 / 14 / 0
- **Avg / median % per leg:** -2.05% / -2.27%
- **Sum % (uncompounded):** -28.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 2 | 14.3% | 0 | 14 | 0 | -2.05% | -28.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 0 | 14 | 0 | -2.05% | -28.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 2 | 14.3% | 0 | 14 | 0 | -2.05% | -28.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1823.00 | 1894.80 | 1894.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 1817.65 | 1873.62 | 1881.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1519.10 | 1516.92 | 1605.60 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 12:15:00 | 1607.40 | 1516.35 | 1587.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 1607.40 | 1516.35 | 1587.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-13 14:15:00 | 1565.80 | 1522.92 | 1587.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-13 15:15:00 | 1567.90 | 1523.36 | 1587.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-19 11:15:00 | 1564.20 | 1537.18 | 1587.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 13:15:00 | 1566.10 | 1537.74 | 1587.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-20 12:15:00 | 1565.30 | 1539.47 | 1586.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:15:00 | 1561.80 | 1539.92 | 1586.47 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-21 15:15:00 | 1566.90 | 1541.91 | 1585.66 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1557.00 | 1542.06 | 1585.51 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-05-23 14:15:00 | 1564.40 | 1544.46 | 1584.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-26 09:15:00 | 1582.30 | 1545.05 | 1584.12 | ENTRY2 sustain failed after 4020m |
| Cross detected — sustain check pending | 2025-05-27 09:15:00 | 1563.90 | 1547.19 | 1583.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-27 10:15:00 | 1568.80 | 1547.41 | 1583.79 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-27 12:15:00 | 1565.00 | 1547.84 | 1583.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-27 14:15:00 | 1569.90 | 1548.23 | 1583.48 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-05-30 09:15:00 | 1561.10 | 1552.63 | 1583.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1563.30 | 1552.82 | 1582.85 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1578.80 | 1553.32 | 1577.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-09 14:15:00 | 1572.90 | 1554.36 | 1577.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-10 09:15:00 | 1585.00 | 1554.84 | 1577.60 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1624.40 | 1559.25 | 1578.74 | SL hit (close>static) qty=1.00 sl=1609.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1624.40 | 1559.25 | 1578.74 | SL hit (close>static) qty=1.00 sl=1609.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1624.40 | 1559.25 | 1578.74 | SL hit (close>static) qty=1.00 sl=1609.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1624.40 | 1559.25 | 1578.74 | SL hit (close>static) qty=1.00 sl=1609.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-14 10:15:00 | 1572.10 | 1604.07 | 1598.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 1561.90 | 1603.30 | 1598.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1601.00 | 1601.95 | 1597.98 | SL hit (close>static) qty=1.00 sl=1588.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 12:15:00 | 1573.10 | 1597.64 | 1596.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 1569.00 | 1597.10 | 1596.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-23 15:15:00 | 1558.90 | 1595.68 | 1595.40 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1556.30 | 1595.28 | 1595.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.30 | 1594.52 | 1594.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.40 | 1495.73 | 1531.88 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1535.40 | 1496.20 | 1529.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.40 | 1496.20 | 1529.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 1506.50 | 1500.07 | 1529.10 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 1504.40 | 1500.21 | 1528.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1509.60 | 1491.77 | 1515.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1511.30 | 1492.14 | 1515.55 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 1506.60 | 1495.35 | 1515.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 1509.70 | 1495.59 | 1515.76 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 1507.50 | 1503.65 | 1516.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 1501.40 | 1503.60 | 1516.83 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1490.30 | 1483.09 | 1500.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 1488.40 | 1486.38 | 1500.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-13 12:15:00 | 1488.80 | 1486.40 | 1500.62 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-10-14 15:15:00 | 1488.00 | 1486.85 | 1500.16 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1473.70 | 1486.72 | 1500.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1541.50 | 1481.50 | 1495.28 | SL hit (close>static) qty=1.00 sl=1535.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-31 10:15:00 | 1486.70 | 1490.57 | 1497.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 1487.10 | 1490.49 | 1497.70 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1518.20 | 1486.45 | 1494.42 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 12:15:00 | 1487.10 | 1498.22 | 1499.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-18 13:15:00 | 1491.60 | 1498.15 | 1499.49 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 1484.50 | 1498.01 | 1499.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-19 09:15:00 | 1527.90 | 1498.20 | 1499.49 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2026-02-11 09:15:00 | 1488.00 | 1598.52 | 1593.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 1484.20 | 1596.29 | 1592.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.90 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1289.20 | 1314.93 | 1375.60 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:15:00 | 1287.20 | 1314.38 | 1374.72 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1271.60 | 1311.16 | 1360.16 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1257.90 | 1310.17 | 1359.18 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-19 13:15:00 | 1566.10 | 2025-06-11 12:15:00 | 1624.40 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-05-20 14:15:00 | 1561.80 | 2025-06-11 12:15:00 | 1624.40 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1557.00 | 2025-06-11 12:15:00 | 1624.40 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-05-30 11:15:00 | 1563.30 | 2025-06-11 12:15:00 | 1624.40 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-07-14 12:15:00 | 1561.90 | 2025-07-15 10:15:00 | 1601.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-07-22 14:15:00 | 1569.00 | 2025-07-24 10:15:00 | 1560.80 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1556.30 | 2025-07-24 10:15:00 | 1560.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-08-28 11:15:00 | 1504.40 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-11 11:15:00 | 1511.30 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-09-15 11:15:00 | 1509.70 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-22 11:15:00 | 1501.40 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-10-15 09:15:00 | 1473.70 | 2025-10-23 10:15:00 | 1541.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-10-31 12:15:00 | 1487.10 | 2025-11-10 10:15:00 | 1518.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-02-11 11:15:00 | 1484.20 | 2026-02-12 10:15:00 | 1405.10 | STOP_HIT | 1.00 | 5.33% |
