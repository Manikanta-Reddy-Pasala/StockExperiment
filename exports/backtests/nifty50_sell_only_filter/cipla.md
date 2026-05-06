# CIPLA (CIPLA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1364.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 10 |
| PENDING | 45 |
| PENDING_CANCEL | 12 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 29
- **Target hits / Stop hits / Partials:** 0 / 33 / 1
- **Avg / median % per leg:** -0.90% / -0.99%
- **Sum % (uncompounded):** -30.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 5 | 14.7% | 0 | 33 | 1 | -0.90% | -30.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 5 | 14.7% | 0 | 33 | 1 | -0.90% | -30.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 5 | 14.7% | 0 | 33 | 1 | -0.90% | -30.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 1442.10 | 1407.43 | 1407.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 1459.00 | 1408.27 | 1407.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1438.65 | 1438.99 | 1426.12 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 1448.30 | 1439.08 | 1426.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1448.30 | 1439.08 | 1426.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-04 14:15:00 | 1465.25 | 1439.52 | 1426.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 1462.55 | 1439.75 | 1426.76 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-09-17 13:15:00 | 1681.93 | 1612.15 | 1574.09 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-10-25 09:15:00 | 1480.60 | 1586.62 | 1586.85 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 1462.55 | 1574.11 | 1580.36 | SL hit qty=0.50 sl=1462.55 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-29 10:15:00 | 1465.65 | 1573.03 | 1579.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 11:15:00 | 1469.70 | 1572.01 | 1579.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 1418.55 | 1566.67 | 1576.36 | SL hit qty=1.00 sl=1418.55 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-31 09:15:00 | 1525.00 | 1557.81 | 1571.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 1554.90 | 1557.78 | 1571.43 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-21 14:15:00 | 1466.00 | 1540.93 | 1558.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 15:15:00 | 1462.20 | 1540.15 | 1557.81 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1486.65 | 1539.62 | 1557.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-22 13:15:00 | 1494.65 | 1537.56 | 1556.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-22 14:15:00 | 1487.10 | 1537.05 | 1555.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-25 09:15:00 | 1494.70 | 1536.13 | 1555.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 1497.30 | 1535.74 | 1554.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-25 13:15:00 | 1496.60 | 1534.54 | 1553.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 1503.00 | 1534.23 | 1553.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-27 14:15:00 | 1493.30 | 1528.64 | 1549.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-11-27 15:15:00 | 1487.00 | 1528.22 | 1549.13 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-28 13:15:00 | 1497.35 | 1526.17 | 1547.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 1495.00 | 1525.86 | 1547.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-05 12:15:00 | 1489.80 | 1523.22 | 1542.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:15:00 | 1496.45 | 1522.96 | 1542.47 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1501.50 | 1522.74 | 1542.27 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 1461.85 | 1514.71 | 1536.41 | SL hit qty=1.00 sl=1461.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 1461.85 | 1514.71 | 1536.41 | SL hit qty=1.00 sl=1461.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 1461.85 | 1514.71 | 1536.41 | SL hit qty=1.00 sl=1461.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 1461.85 | 1514.71 | 1536.41 | SL hit qty=1.00 sl=1461.85 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-19 12:15:00 | 1505.95 | 1492.33 | 1518.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:15:00 | 1508.40 | 1492.49 | 1518.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 1485.25 | 1492.74 | 1518.17 | SL hit qty=1.00 sl=1485.25 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-27 10:15:00 | 1507.30 | 1489.96 | 1513.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-27 11:15:00 | 1503.05 | 1490.09 | 1513.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-27 12:15:00 | 1508.65 | 1490.28 | 1513.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 1509.50 | 1490.47 | 1513.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 1485.25 | 1501.35 | 1515.46 | SL hit qty=1.00 sl=1485.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-07 10:15:00 | 1506.90 | 1500.69 | 1514.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-07 11:15:00 | 1497.55 | 1500.66 | 1514.48 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 1418.55 | 1470.38 | 1491.50 | SL hit qty=1.00 sl=1418.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 1418.55 | 1470.38 | 1491.50 | SL hit qty=1.00 sl=1418.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-18 10:15:00 | 1507.50 | 1456.56 | 1464.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 11:15:00 | 1507.95 | 1457.07 | 1465.17 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-18 14:15:00 | 1510.05 | 1458.53 | 1465.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 15:15:00 | 1508.65 | 1459.03 | 1466.00 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 1443.55 | 1472.34 | 1472.45 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1495.95 | 1472.68 | 1472.60 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-04-04 09:15:00 | 1403.70 | 1472.22 | 1472.37 | slope filter: EMA200 not falling 0.50% over 350 bars |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 1500.30 | 1461.15 | 1465.59 | Break + close above crossover candle high |

### Cycle 5 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.84 | 1469.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1476.00 | 1472.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.98 | 1488.64 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 1488.40 | 1502.18 | 1489.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1488.40 | 1502.18 | 1489.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 10:15:00 | 1503.70 | 1499.66 | 1488.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 1505.20 | 1499.72 | 1488.92 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 1483.20 | 1501.53 | 1490.48 | SL hit qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-14 14:15:00 | 1493.90 | 1501.24 | 1490.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 1499.90 | 1501.22 | 1490.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 1483.20 | 1500.99 | 1490.58 | SL hit qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 1506.00 | 1500.48 | 1490.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 1499.80 | 1500.47 | 1490.57 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 1483.20 | 1499.72 | 1490.95 | SL hit qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-05 12:15:00 | 1495.70 | 1485.99 | 1485.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-05 13:15:00 | 1488.10 | 1486.01 | 1485.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 1495.10 | 1486.21 | 1485.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-06 11:15:00 | 1492.80 | 1486.28 | 1485.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 12:15:00 | 1496.90 | 1486.38 | 1485.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:15:00 | 1498.00 | 1486.50 | 1485.99 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1503.90 | 1493.09 | 1489.60 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1483.20 | 1493.09 | 1489.60 | SL hit qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 09:15:00 | 1535.50 | 1494.03 | 1490.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 1530.80 | 1494.39 | 1490.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-17 13:15:00 | 1510.30 | 1496.72 | 1491.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-17 14:15:00 | 1503.60 | 1496.79 | 1491.85 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1482.30 | 1496.57 | 1492.10 | SL hit qty=1.00 sl=1482.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 1512.60 | 1496.92 | 1492.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 1511.00 | 1497.06 | 1492.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-25 10:15:00 | 1510.30 | 1497.68 | 1493.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-25 11:15:00 | 1501.90 | 1497.72 | 1493.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-25 14:15:00 | 1510.20 | 1498.04 | 1493.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 1509.80 | 1498.16 | 1493.52 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-26 13:15:00 | 1510.00 | 1498.77 | 1493.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 1513.50 | 1498.92 | 1494.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1498.80 | 1499.06 | 1494.16 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-27 15:15:00 | 1506.30 | 1499.12 | 1494.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1506.40 | 1499.20 | 1494.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-06-30 11:15:00 | 1505.00 | 1499.28 | 1494.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 1507.20 | 1499.36 | 1494.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-01 12:15:00 | 1507.80 | 1499.72 | 1494.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1514.80 | 1499.87 | 1495.00 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-03 11:15:00 | 1510.00 | 1500.61 | 1495.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1508.50 | 1500.69 | 1495.73 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1493.20 | 1502.35 | 1497.03 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1492.10 | 1502.35 | 1497.03 | SL hit qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1492.10 | 1502.35 | 1497.03 | SL hit qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1492.10 | 1502.35 | 1497.03 | SL hit qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1492.10 | 1502.35 | 1497.03 | SL hit qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1482.30 | 1502.14 | 1496.95 | SL hit qty=1.00 sl=1482.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1482.30 | 1502.14 | 1496.95 | SL hit qty=1.00 sl=1482.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1482.30 | 1502.14 | 1496.95 | SL hit qty=1.00 sl=1482.30 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-18 13:15:00 | 1475.30 | 1493.13 | 1493.18 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-07-25 13:15:00 | 1541.60 | 1488.41 | 1490.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:15:00 | 1532.60 | 1488.85 | 1490.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 1567.30 | 1492.86 | 1492.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.30 | 1492.86 | 1492.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.40 | 1493.66 | 1493.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1507.00 | 1507.11 | 1500.40 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 1504.00 | 1507.04 | 1500.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1504.00 | 1507.04 | 1500.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-04 09:15:00 | 1512.30 | 1506.96 | 1500.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:15:00 | 1511.10 | 1507.00 | 1500.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-05 10:15:00 | 1506.70 | 1507.36 | 1500.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-05 11:15:00 | 1504.40 | 1507.33 | 1500.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 1499.20 | 1507.33 | 1500.97 | SL hit qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-12 09:15:00 | 1513.10 | 1502.32 | 1499.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1507.60 | 1502.37 | 1499.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1499.20 | 1550.65 | 1539.59 | SL hit qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 1520.90 | 1547.27 | 1538.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 1519.00 | 1546.99 | 1538.15 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 1499.20 | 1545.64 | 1537.61 | SL hit qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 11:15:00 | 1505.60 | 1540.30 | 1535.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-01 12:15:00 | 1503.10 | 1539.93 | 1535.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-01 13:15:00 | 1515.40 | 1539.68 | 1535.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:15:00 | 1513.70 | 1539.43 | 1534.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1515.00 | 1539.18 | 1534.85 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 1518.30 | 1538.97 | 1534.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 10:15:00 | 1510.00 | 1538.69 | 1534.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-03 11:15:00 | 1518.70 | 1538.49 | 1534.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 1518.50 | 1538.29 | 1534.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-03 14:15:00 | 1519.00 | 1537.85 | 1534.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 1517.70 | 1537.65 | 1534.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1499.20 | 1537.31 | 1534.06 | SL hit qty=1.00 sl=1499.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1510.80 | 1537.31 | 1534.06 | SL hit qty=1.00 sl=1510.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1510.80 | 1537.31 | 1534.06 | SL hit qty=1.00 sl=1510.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 1517.50 | 1535.67 | 1533.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 10:15:00 | 1516.40 | 1535.47 | 1533.26 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 1518.80 | 1535.06 | 1533.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 1517.50 | 1534.88 | 1532.99 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1510.80 | 1534.11 | 1532.63 | SL hit qty=1.00 sl=1510.80 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-09 12:15:00 | 1513.00 | 1531.21 | 1531.21 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 1520.20 | 1530.57 | 1530.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1539.00 | 1530.65 | 1530.93 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1555.20 | 1531.40 | 1531.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.40 | 1531.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1561.80 | 1531.98 | 1531.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-11-12 10:15:00 | 1525.00 | 1539.70 | 1539.76 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 15:15:00 | 1462.55 | 2024-09-17 13:15:00 | 1681.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-04 15:15:00 | 1462.55 | 2024-10-29 09:15:00 | 1462.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-10-29 11:15:00 | 1469.70 | 2024-10-30 09:15:00 | 1418.55 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-10-31 10:15:00 | 1554.90 | 2024-12-10 11:15:00 | 1461.85 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2024-11-21 15:15:00 | 1462.20 | 2024-12-10 11:15:00 | 1461.85 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-11-25 10:15:00 | 1497.30 | 2024-12-10 11:15:00 | 1461.85 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-11-25 14:15:00 | 1503.00 | 2024-12-10 11:15:00 | 1461.85 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-11-28 14:15:00 | 1495.00 | 2024-12-20 12:15:00 | 1485.25 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-12-05 13:15:00 | 1496.45 | 2025-01-06 09:15:00 | 1485.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-19 13:15:00 | 1508.40 | 2025-01-24 13:15:00 | 1418.55 | STOP_HIT | 1.00 | -5.96% |
| BUY | retest2 | 2024-12-27 13:15:00 | 1509.50 | 2025-01-24 13:15:00 | 1418.55 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest2 | 2025-03-18 11:15:00 | 1507.95 | 2025-03-24 14:15:00 | 1524.50 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-03-18 15:15:00 | 1508.65 | 2025-03-24 14:15:00 | 1524.50 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-05-12 11:15:00 | 1505.20 | 2025-05-14 09:15:00 | 1483.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-14 15:15:00 | 1499.90 | 2025-05-15 09:15:00 | 1483.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-15 14:15:00 | 1499.80 | 2025-05-20 09:15:00 | 1483.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-06 13:15:00 | 1498.00 | 2025-06-13 09:15:00 | 1483.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-16 10:15:00 | 1530.80 | 2025-06-19 15:15:00 | 1482.30 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1511.00 | 2025-07-08 09:15:00 | 1492.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-25 15:15:00 | 1509.80 | 2025-07-08 09:15:00 | 1492.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-26 14:15:00 | 1513.50 | 2025-07-08 09:15:00 | 1492.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1506.40 | 2025-07-08 09:15:00 | 1492.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-30 12:15:00 | 1507.20 | 2025-07-08 10:15:00 | 1482.30 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-01 13:15:00 | 1514.80 | 2025-07-08 10:15:00 | 1482.30 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-07-03 12:15:00 | 1508.50 | 2025-07-08 10:15:00 | 1482.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-25 14:15:00 | 1532.60 | 2025-07-28 13:15:00 | 1567.30 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2025-08-04 10:15:00 | 1511.10 | 2025-08-05 11:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-12 10:15:00 | 1507.60 | 2025-09-26 09:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-29 10:15:00 | 1519.00 | 2025-09-29 13:15:00 | 1499.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-01 14:15:00 | 1513.70 | 2025-10-06 09:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-03 12:15:00 | 1518.50 | 2025-10-06 09:15:00 | 1510.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-03 15:15:00 | 1517.70 | 2025-10-06 09:15:00 | 1510.80 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-10-07 13:15:00 | 1517.50 | 2025-10-08 09:15:00 | 1510.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1539.00 | 2025-10-10 13:15:00 | 1555.20 | STOP_HIT | 1.00 | 1.05% |
