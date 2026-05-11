# HCL Technologies Ltd. (HCLTECH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 21 |
| PARTIAL | 9 |
| TARGET_HIT | 17 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 5
- **Target hits / Stop hits / Partials:** 17 / 5 / 9
- **Avg / median % per leg:** 6.54% / 9.14%
- **Sum % (uncompounded):** 202.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 8 | 3 | 0 | 6.61% | 72.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 8 | 72.7% | 8 | 3 | 0 | 6.61% | 72.7% |
| SELL (all) | 20 | 18 | 90.0% | 9 | 2 | 9 | 6.51% | 130.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.54% | -2.5% |
| SELL @ 3rd Alert (retest2) | 19 | 18 | 94.7% | 9 | 1 | 9 | 6.98% | 132.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.54% | -2.5% |
| retest2 (combined) | 30 | 26 | 86.7% | 17 | 4 | 9 | 6.84% | 205.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1650.80 | 1606.38 | 1606.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1668.90 | 1622.83 | 1615.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.70 | 1692.97 | 1665.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.37 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.50 | 1646.92 | 1648.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.10 | 1476.51 | 1514.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1444.30 | 1442.23 | 1479.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1477.40 | 1443.05 | 1478.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1481.00 | 1443.42 | 1478.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 1481.00 | 1443.42 | 1478.65 | SL hit (close>ema400) qty=1.00 sl=1478.65 alert=retest1 |

### Cycle 3 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.73 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.17 | 1495.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 1501.60 | 1501.08 | 1498.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1508.90 | 1501.16 | 1498.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 1510.40 | 1501.16 | 1498.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 1510.10 | 1501.31 | 1498.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 1511.10 | 1501.31 | 1498.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 1510.70 | 1501.40 | 1498.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-19 11:15:00 | 1661.44 | 1536.80 | 1518.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:45:00 | 1395.90 | 1395.18 | 1457.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1437.90 | 1399.27 | 1454.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 1432.60 | 1399.27 | 1454.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 1465.70 | 1405.31 | 1454.03 | SL hit (close>static) qty=1.00 sl=1464.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-10-08 09:15:00 | 1444.30 | 2025-10-09 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-07 11:15:00 | 1510.40 | 2025-11-19 11:15:00 | 1661.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 12:30:00 | 1510.10 | 2025-11-19 11:15:00 | 1661.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 13:15:00 | 1511.10 | 2025-11-19 12:15:00 | 1662.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 14:00:00 | 1510.70 | 2025-11-19 12:15:00 | 1661.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 12:15:00 | 1608.90 | 2026-02-03 09:15:00 | 1769.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 13:15:00 | 1608.30 | 2026-02-03 09:15:00 | 1769.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 14:45:00 | 1610.10 | 2026-02-03 09:15:00 | 1771.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1611.90 | 2026-02-03 09:15:00 | 1773.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1636.60 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1642.00 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-05 09:45:00 | 1625.10 | 2026-02-06 09:15:00 | 1594.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-04-08 10:15:00 | 1432.60 | 2026-04-09 14:15:00 | 1465.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 09:15:00 | 1358.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 09:15:00 | 1362.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 09:15:00 | 1350.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 09:15:00 | 1368.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 09:15:00 | 1369.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 09:15:00 | 1371.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 09:15:00 | 1364.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-22 09:15:00 | 1353.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 09:30:00 | 1430.00 | 2026-04-22 10:15:00 | 1299.24 | TARGET_HIT | 0.50 | 9.14% |
| SELL | retest2 | 2026-04-10 10:00:00 | 1433.70 | 2026-04-22 11:15:00 | 1290.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1421.80 | 2026-04-22 11:15:00 | 1296.09 | TARGET_HIT | 0.50 | 8.84% |
| SELL | retest2 | 2026-04-15 12:45:00 | 1440.10 | 2026-04-22 11:15:00 | 1297.44 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1441.60 | 2026-04-22 11:15:00 | 1293.03 | TARGET_HIT | 0.50 | 10.31% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1443.60 | 2026-04-22 12:15:00 | 1287.00 | TARGET_HIT | 0.50 | 10.85% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1436.70 | 2026-04-22 12:15:00 | 1281.96 | TARGET_HIT | 0.50 | 10.77% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1424.40 | 2026-04-23 09:15:00 | 1279.62 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-04-24 09:15:00 | 1242.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1308.00 | 2026-05-08 09:15:00 | 1177.20 | TARGET_HIT | 0.50 | 10.00% |
