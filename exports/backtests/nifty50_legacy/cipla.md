# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1347.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 9 |
| ALERT3 | 13 |
| PENDING | 34 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 20
- **Target hits / Stop hits / Partials:** 0 / 25 / 4
- **Avg / median % per leg:** 2.53% / -1.12%
- **Sum % (uncompounded):** 73.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 2 | 9.5% | 0 | 20 | 1 | -0.22% | -4.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 2 | 9.5% | 0 | 20 | 1 | -0.22% | -4.7% |
| SELL (all) | 8 | 7 | 87.5% | 0 | 5 | 3 | 9.75% | 78.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 7 | 87.5% | 0 | 5 | 3 | 9.75% | 78.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 9 | 31.0% | 0 | 25 | 4 | 2.53% | 73.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 12:15:00 | 1375.50 | 1407.34 | 1407.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 13:15:00 | 1369.50 | 1406.96 | 1407.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 1418.75 | 1406.14 | 1406.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 1418.75 | 1406.14 | 1406.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1418.75 | 1406.14 | 1406.83 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 1442.10 | 1407.43 | 1407.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 1459.00 | 1408.27 | 1407.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1438.65 | 1438.99 | 1426.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 1448.30 | 1439.08 | 1426.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1448.30 | 1439.08 | 1426.23 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-04 14:15:00 | 1465.25 | 1439.52 | 1426.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 1462.55 | 1439.75 | 1426.76 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 13:15:00 | 1681.93 | 1612.15 | 1574.09 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1619.75 | 1623.27 | 1587.16 | SL hit (close<ema200) qty=0.50 sl=1623.27 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1480.60 | 1586.62 | 1586.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1457.85 | 1574.11 | 1580.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1559.80 | 1557.80 | 1571.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1576.80 | 1557.97 | 1571.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1576.80 | 1557.97 | 1571.12 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-11 11:15:00 | 1553.00 | 1566.20 | 1573.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 1549.70 | 1566.04 | 1573.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 1524.50 | 1472.32 | 1472.17 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 1443.55 | 1472.34 | 1472.45 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1495.95 | 1472.68 | 1472.60 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1403.70 | 1472.22 | 1472.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1467.97 | 1470.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1467.80 | 1457.10 | 1464.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1467.80 | 1457.10 | 1464.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1467.80 | 1457.10 | 1464.21 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-04-11 10:15:00 | 1454.50 | 1457.08 | 1464.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:15:00 | 1456.75 | 1457.07 | 1464.13 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 1494.60 | 1458.72 | 1464.55 | SL hit (close>static) qty=1.00 sl=1488.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.84 | 1469.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1476.00 | 1472.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.98 | 1488.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 1488.40 | 1502.18 | 1489.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1488.40 | 1502.18 | 1489.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-12 10:15:00 | 1503.70 | 1499.66 | 1488.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 1505.20 | 1499.72 | 1488.92 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 14:15:00 | 1493.90 | 1501.24 | 1490.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 1499.90 | 1501.22 | 1490.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 1477.30 | 1500.99 | 1490.58 | SL hit (close<static) qty=1.00 sl=1483.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 1477.30 | 1500.99 | 1490.58 | SL hit (close<static) qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 1506.00 | 1500.48 | 1490.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 1499.80 | 1500.47 | 1490.57 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 1471.30 | 1499.72 | 1490.95 | SL hit (close<static) qty=1.00 sl=1483.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-05 12:15:00 | 1495.70 | 1485.99 | 1485.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-05 13:15:00 | 1488.10 | 1486.01 | 1485.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 1495.10 | 1486.21 | 1485.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-06 11:15:00 | 1492.80 | 1486.28 | 1485.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 12:15:00 | 1496.90 | 1486.38 | 1485.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:15:00 | 1498.00 | 1486.50 | 1485.99 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1503.90 | 1493.09 | 1489.60 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-16 09:15:00 | 1535.50 | 1494.03 | 1490.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 1530.80 | 1494.39 | 1490.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-17 13:15:00 | 1510.30 | 1496.72 | 1491.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-17 14:15:00 | 1503.60 | 1496.79 | 1491.85 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 1483.00 | 1496.74 | 1492.16 | SL hit (close<static) qty=1.00 sl=1483.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1480.50 | 1496.57 | 1492.10 | SL hit (close<static) qty=1.00 sl=1482.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 1512.60 | 1496.92 | 1492.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 1511.00 | 1497.06 | 1492.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-25 10:15:00 | 1510.30 | 1497.68 | 1493.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-25 11:15:00 | 1501.90 | 1497.72 | 1493.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-25 14:15:00 | 1510.20 | 1498.04 | 1493.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 1509.80 | 1498.16 | 1493.52 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-26 13:15:00 | 1510.00 | 1498.77 | 1493.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 1513.50 | 1498.92 | 1494.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1498.80 | 1499.06 | 1494.16 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-27 15:15:00 | 1506.30 | 1499.12 | 1494.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1506.40 | 1499.20 | 1494.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-06-30 11:15:00 | 1505.00 | 1499.28 | 1494.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 1507.20 | 1499.36 | 1494.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-01 12:15:00 | 1507.80 | 1499.72 | 1494.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1514.80 | 1499.87 | 1495.00 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-03 11:15:00 | 1510.00 | 1500.61 | 1495.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1508.50 | 1500.69 | 1495.73 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1493.20 | 1502.35 | 1497.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1482.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1482.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1482.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1492.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.50 | 1502.14 | 1496.95 | SL hit (close<static) qty=1.00 sl=1492.10 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1475.30 | 1493.13 | 1493.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1472.20 | 1492.26 | 1492.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 1488.20 | 1488.02 | 1490.42 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-07-25 09:15:00 | 1482.00 | 1487.95 | 1490.36 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-25 10:15:00 | 1485.00 | 1487.92 | 1490.33 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 12:15:00 | 1486.60 | 1487.88 | 1490.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1486.60 | 1487.88 | 1490.29 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.30 | 1492.86 | 1492.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.40 | 1493.66 | 1493.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1507.00 | 1507.11 | 1500.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 1504.00 | 1507.04 | 1500.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1504.00 | 1507.04 | 1500.43 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-08-04 09:15:00 | 1512.30 | 1506.96 | 1500.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:15:00 | 1511.10 | 1507.00 | 1500.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-05 10:15:00 | 1506.70 | 1507.36 | 1500.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-05 11:15:00 | 1504.40 | 1507.33 | 1500.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1494.80 | 1507.21 | 1500.94 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-12 09:15:00 | 1513.10 | 1502.32 | 1499.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1507.60 | 1502.37 | 1499.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1496.70 | 1550.65 | 1539.59 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 1520.90 | 1547.27 | 1538.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 1519.00 | 1546.99 | 1538.15 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 1496.40 | 1545.64 | 1537.61 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 11:15:00 | 1505.60 | 1540.30 | 1535.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-01 12:15:00 | 1503.10 | 1539.93 | 1535.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-01 13:15:00 | 1515.40 | 1539.68 | 1535.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:15:00 | 1513.70 | 1539.43 | 1534.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1515.00 | 1539.18 | 1534.85 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 1518.30 | 1538.97 | 1534.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 10:15:00 | 1510.00 | 1538.69 | 1534.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-03 11:15:00 | 1518.70 | 1538.49 | 1534.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 1518.50 | 1538.29 | 1534.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-03 14:15:00 | 1519.00 | 1537.85 | 1534.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 1517.70 | 1537.65 | 1534.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1503.00 | 1537.31 | 1534.06 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1503.00 | 1537.31 | 1534.06 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 1517.50 | 1535.67 | 1533.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 10:15:00 | 1516.40 | 1535.47 | 1533.26 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 1518.80 | 1535.06 | 1533.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 1517.50 | 1534.88 | 1532.99 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1500.70 | 1534.11 | 1532.63 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1495.00 | 1532.39 | 1531.79 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.21 | 1531.21 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.40 | 1531.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1561.80 | 1531.98 | 1531.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.46 | 1549.25 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1525.00 | 1539.70 | 1539.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.13 | 1539.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.20 | 1529.77 | 1533.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1532.20 | 1529.80 | 1533.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1532.20 | 1529.80 | 1533.78 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-28 15:15:00 | 1529.00 | 1529.79 | 1533.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1524.60 | 1529.74 | 1533.71 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-12-02 12:15:00 | 1516.80 | 1529.21 | 1533.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 1519.90 | 1529.12 | 1533.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 1503.00 | 1511.12 | 1517.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1480.10 | 1510.81 | 1517.75 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1295.91 | 1450.86 | 1480.92 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1291.92 | 1450.86 | 1480.92 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1362.20 | 1359.14 | 1401.43 | SL hit (close>ema200) qty=0.50 sl=1359.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1362.20 | 1359.14 | 1401.43 | SL hit (close>ema200) qty=0.50 sl=1359.14 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:15:00 | 1258.08 | 1330.96 | 1368.65 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1289.10 | 1251.00 | 1295.53 | SL hit (close>ema200) qty=0.50 sl=1251.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 15:15:00 | 1462.55 | 2024-09-17 13:15:00 | 1681.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-04 15:15:00 | 1462.55 | 2024-09-25 11:15:00 | 1619.75 | STOP_HIT | 0.50 | 10.75% |
| SELL | retest2 | 2024-11-11 12:15:00 | 1549.70 | 2025-03-24 14:15:00 | 1524.50 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-04-11 11:15:00 | 1456.75 | 2025-04-16 09:15:00 | 1494.60 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-05-12 11:15:00 | 1505.20 | 2025-05-15 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-05-14 15:15:00 | 1499.90 | 2025-05-15 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-15 14:15:00 | 1499.80 | 2025-05-20 09:15:00 | 1471.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-06-06 13:15:00 | 1498.00 | 2025-06-19 14:15:00 | 1483.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-16 10:15:00 | 1530.80 | 2025-06-19 15:15:00 | 1480.50 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1511.00 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-25 15:15:00 | 1509.80 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-26 14:15:00 | 1513.50 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1506.40 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-30 12:15:00 | 1507.20 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-07-01 13:15:00 | 1514.80 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-07-03 12:15:00 | 1508.50 | 2025-07-08 10:15:00 | 1481.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-04 10:15:00 | 1511.10 | 2025-08-05 12:15:00 | 1494.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-12 10:15:00 | 1507.60 | 2025-09-26 09:15:00 | 1496.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-29 10:15:00 | 1519.00 | 2025-09-29 13:15:00 | 1496.40 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-01 14:15:00 | 1513.70 | 2025-10-06 09:15:00 | 1503.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-03 12:15:00 | 1518.50 | 2025-10-06 09:15:00 | 1503.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-03 15:15:00 | 1517.70 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-07 13:15:00 | 1517.50 | 2025-10-08 14:15:00 | 1495.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-01-27 09:15:00 | 1295.91 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-12-02 13:15:00 | 1519.90 | 2026-01-27 09:15:00 | 1291.92 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-02-26 09:15:00 | 1362.20 | STOP_HIT | 0.50 | 10.65% |
| SELL | retest2 | 2025-12-02 13:15:00 | 1519.90 | 2026-02-26 09:15:00 | 1362.20 | STOP_HIT | 0.50 | 10.38% |
| SELL | retest2 | 2026-01-07 10:15:00 | 1480.10 | 2026-03-19 09:15:00 | 1258.08 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 1480.10 | 2026-04-23 09:15:00 | 1289.10 | STOP_HIT | 0.50 | 12.90% |
