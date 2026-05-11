# INFY (INFY)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1179.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 2 / 15 / 5
- **Avg / median % per leg:** 1.37% / 0.41%
- **Sum % (uncompounded):** 30.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 11 | 50.0% | 2 | 15 | 5 | 1.37% | 30.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 0 | 15 | 3 | 0.01% | 0.2% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 18 | 7 | 38.9% | 0 | 15 | 3 | 0.01% | 0.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1492.00 | 1568.53 | 1568.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1481.10 | 1566.87 | 1567.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.40 | 1495.70 | 1524.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:45:00 | 1497.30 | 1495.70 | 1524.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1521.90 | 1495.78 | 1522.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 1527.40 | 1495.78 | 1522.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.40 | 1496.17 | 1522.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 1532.40 | 1496.17 | 1522.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1534.90 | 1496.56 | 1522.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1534.90 | 1496.56 | 1522.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1531.70 | 1498.69 | 1522.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 1534.20 | 1498.69 | 1522.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 1525.50 | 1499.22 | 1522.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 1525.50 | 1499.22 | 1522.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1521.60 | 1499.44 | 1522.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 1525.20 | 1499.44 | 1522.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1527.80 | 1499.72 | 1522.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1527.80 | 1499.72 | 1522.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1525.80 | 1499.98 | 1522.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1513.20 | 1499.98 | 1522.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 1437.54 | 1494.31 | 1515.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1496.70 | 1488.06 | 1510.53 | SL hit (close>ema200) qty=0.50 sl=1488.06 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 1538.00 | 1498.99 | 1498.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1546.40 | 1503.83 | 1501.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 1591.10 | 1606.85 | 1574.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 1605.90 | 1634.65 | 1606.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1550.60 | 1635.99 | 1609.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.87 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1288.70 | 1314.93 | 1375.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1278.60 | 1311.56 | 1360.58 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1224.26 | 1303.24 | 1352.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1214.67 | 1303.24 | 1352.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-24 13:15:00 | 1159.83 | 1297.91 | 1349.07 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-05 10:15:00 | 1437.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-09 10:15:00 | 1496.70 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-09-10 09:45:00 | 1522.80 | 2025-09-17 09:15:00 | 1516.60 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1516.80 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-12 09:30:00 | 1524.00 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-15 15:00:00 | 1507.30 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1503.70 | 2025-10-01 09:15:00 | 1428.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:00:00 | 1507.50 | 2025-10-01 09:15:00 | 1432.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1503.70 | 2025-10-08 09:15:00 | 1486.40 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-09-22 10:00:00 | 1507.50 | 2025-10-08 09:15:00 | 1486.40 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-10-09 13:00:00 | 1504.80 | 2025-10-10 09:15:00 | 1516.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-15 09:15:00 | 1486.40 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-10-31 11:00:00 | 1486.70 | 2025-11-10 10:15:00 | 1518.20 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-10-31 12:00:00 | 1486.00 | 2025-11-10 10:15:00 | 1518.20 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-10-31 13:15:00 | 1484.80 | 2025-11-10 10:15:00 | 1518.20 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1500.40 | 2025-11-17 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-17 13:00:00 | 1502.30 | 2025-11-17 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1498.30 | 2025-11-19 09:15:00 | 1527.90 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest1 | 2026-04-10 09:30:00 | 1288.70 | 2026-04-24 09:15:00 | 1224.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 1278.60 | 2026-04-24 09:15:00 | 1214.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-10 09:30:00 | 1288.70 | 2026-04-24 13:15:00 | 1159.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 1278.60 | 2026-04-28 12:15:00 | 1150.74 | TARGET_HIT | 0.50 | 10.00% |
