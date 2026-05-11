# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1253.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 24
- **Target hits / Stop hits / Partials:** 8 / 27 / 13
- **Avg / median % per leg:** 2.53% / 1.50%
- **Sum % (uncompounded):** 121.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.99% | -6.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.99% | -6.0% |
| SELL (all) | 42 | 24 | 57.1% | 8 | 21 | 13 | 3.04% | 127.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 24 | 57.1% | 8 | 21 | 13 | 3.04% | 127.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 24 | 50.0% | 8 | 27 | 13 | 2.53% | 121.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1500.40 | 1558.72 | 1558.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1497.80 | 1558.12 | 1558.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 1528.50 | 1551.33 | 1554.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:45:00 | 1529.60 | 1550.68 | 1553.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 1554.20 | 1550.67 | 1553.67 | SL hit (close>static) qty=1.00 sl=1553.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1585.60 | 1554.72 | 1554.63 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1525.60 | 1554.40 | 1554.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 1515.00 | 1553.26 | 1553.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 1556.40 | 1541.25 | 1546.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1526.40 | 1541.10 | 1546.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:15:00 | 1545.10 | 1541.10 | 1546.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1554.10 | 1541.23 | 1546.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1554.10 | 1541.23 | 1546.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1571.30 | 1541.53 | 1546.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1571.40 | 1541.53 | 1546.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1578.80 | 1541.90 | 1546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1578.80 | 1541.90 | 1546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1552.70 | 1544.64 | 1548.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1552.70 | 1544.64 | 1548.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1547.40 | 1544.67 | 1548.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 1538.90 | 1544.60 | 1548.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1461.95 | 1521.12 | 1533.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1546.70 | 1509.71 | 1525.60 | SL hit (close>ema200) qty=0.50 sl=1509.71 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1572.00 | 1536.20 | 1536.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1581.50 | 1536.65 | 1536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1545.90 | 1567.87 | 1555.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1548.00 | 1567.68 | 1555.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1548.00 | 1567.68 | 1555.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1549.50 | 1566.32 | 1555.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 1549.50 | 1566.32 | 1555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1556.50 | 1566.22 | 1555.47 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1473.00 | 1546.77 | 1546.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1468.50 | 1530.64 | 1537.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1510.60 | 1508.94 | 1523.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:00:00 | 1510.60 | 1508.94 | 1523.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1449.20 | 1427.60 | 1447.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1449.20 | 1427.60 | 1447.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1455.70 | 1427.88 | 1447.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 1455.40 | 1427.88 | 1447.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1456.60 | 1428.17 | 1447.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 1451.90 | 1428.69 | 1447.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1462.50 | 1429.52 | 1447.89 | SL hit (close>static) qty=1.00 sl=1460.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:15:00 | 1570.20 | 2025-05-28 09:15:00 | 1556.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-22 15:00:00 | 1566.00 | 2025-05-28 09:15:00 | 1556.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1568.40 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-05-23 10:30:00 | 1567.70 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-27 14:45:00 | 1573.80 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-27 15:15:00 | 1572.60 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-13 11:30:00 | 1528.50 | 2025-06-16 12:15:00 | 1554.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-06-16 09:45:00 | 1529.60 | 2025-06-16 12:15:00 | 1554.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-17 15:00:00 | 1530.80 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-19 10:30:00 | 1522.00 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-06-23 11:15:00 | 1537.90 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-30 10:15:00 | 1540.00 | 2025-06-30 15:15:00 | 1552.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1539.40 | 2025-06-30 15:15:00 | 1552.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-02 09:30:00 | 1538.30 | 2025-07-02 15:15:00 | 1552.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-25 09:30:00 | 1538.90 | 2025-08-11 09:15:00 | 1461.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 1538.90 | 2025-08-18 09:15:00 | 1546.70 | STOP_HIT | 0.50 | -0.51% |
| SELL | retest2 | 2025-08-18 10:15:00 | 1541.30 | 2025-08-18 10:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1542.60 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1545.00 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-26 11:45:00 | 1535.20 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-26 12:45:00 | 1535.10 | 2025-09-01 12:15:00 | 1561.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-08-26 14:30:00 | 1531.20 | 2025-09-01 12:15:00 | 1561.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-05 14:00:00 | 1451.90 | 2026-01-06 09:15:00 | 1462.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1451.80 | 2026-01-12 14:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1446.60 | 2026-01-20 11:15:00 | 1379.21 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-12 12:15:00 | 1447.30 | 2026-01-20 11:15:00 | 1374.27 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-01-12 13:15:00 | 1443.40 | 2026-01-20 11:15:00 | 1374.93 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1441.70 | 2026-01-20 11:15:00 | 1369.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1435.70 | 2026-01-20 11:15:00 | 1363.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1436.00 | 2026-01-20 11:15:00 | 1364.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1446.60 | 2026-01-21 09:15:00 | 1306.62 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2026-01-12 12:15:00 | 1447.30 | 2026-01-21 09:15:00 | 1301.94 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2026-01-12 13:15:00 | 1443.40 | 2026-01-21 09:15:00 | 1302.57 | TARGET_HIT | 0.50 | 9.76% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1441.70 | 2026-01-21 09:15:00 | 1297.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 10:15:00 | 1399.00 | 2026-01-21 09:15:00 | 1329.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1435.70 | 2026-01-23 11:15:00 | 1292.40 | TARGET_HIT | 0.50 | 9.98% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1436.00 | 2026-01-23 12:15:00 | 1292.13 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2026-01-20 10:15:00 | 1399.00 | 2026-01-29 10:15:00 | 1259.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1400.00 | 2026-03-04 09:15:00 | 1330.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1401.20 | 2026-03-04 09:15:00 | 1331.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 1401.80 | 2026-03-04 09:15:00 | 1331.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1372.20 | 2026-03-09 09:15:00 | 1303.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1400.00 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.50% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1401.20 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2026-02-20 11:00:00 | 1401.80 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1372.20 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2026-03-11 09:45:00 | 1387.70 | 2026-03-13 09:15:00 | 1318.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:45:00 | 1387.70 | 2026-03-23 09:15:00 | 1248.93 | TARGET_HIT | 0.50 | 10.00% |
