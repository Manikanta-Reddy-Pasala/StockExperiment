# Infosys Ltd. (INFY)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1179.50
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
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 21
- **Target hits / Stop hits / Partials:** 0 / 22 / 1
- **Avg / median % per leg:** -0.99% / -0.97%
- **Sum % (uncompounded):** -22.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 2 | 8.7% | 0 | 22 | 1 | -0.99% | -22.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 0 | 22 | 1 | -0.99% | -22.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 2 | 8.7% | 0 | 22 | 1 | -0.99% | -22.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 1605.50 | 1591.48 | 1591.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.90 | 1592.54 | 1592.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1594.20 | 1605.23 | 1599.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1590.80 | 1605.08 | 1599.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 1591.10 | 1605.08 | 1599.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1593.80 | 1604.77 | 1599.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 1594.50 | 1604.77 | 1599.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1590.70 | 1601.84 | 1597.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1590.70 | 1601.84 | 1597.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1587.10 | 1601.70 | 1597.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:45:00 | 1586.20 | 1601.70 | 1597.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1591.30 | 1601.04 | 1597.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 1591.90 | 1601.04 | 1597.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1596.00 | 1601.24 | 1597.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1596.00 | 1601.24 | 1597.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1595.00 | 1601.18 | 1597.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1595.00 | 1601.18 | 1597.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1588.50 | 1601.01 | 1597.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1588.50 | 1601.01 | 1597.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1588.20 | 1600.26 | 1597.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 1580.20 | 1600.26 | 1597.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.30 | 1594.52 | 1594.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.40 | 1495.73 | 1531.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:45:00 | 1497.30 | 1495.73 | 1531.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.40 | 1496.20 | 1529.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 1532.40 | 1496.20 | 1529.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1534.90 | 1496.58 | 1529.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1534.90 | 1496.58 | 1529.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1533.30 | 1498.07 | 1529.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1532.20 | 1498.07 | 1529.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1531.70 | 1498.72 | 1529.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 1534.20 | 1498.72 | 1529.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1524.60 | 1498.98 | 1529.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:30:00 | 1533.20 | 1498.98 | 1529.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1527.80 | 1499.75 | 1529.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1527.80 | 1499.75 | 1529.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1525.80 | 1500.00 | 1529.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1513.20 | 1500.00 | 1529.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 1437.54 | 1494.33 | 1520.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1496.70 | 1488.08 | 1515.43 | SL hit (close>ema200) qty=0.50 sl=1488.08 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 1522.80 | 1489.08 | 1515.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1516.80 | 1491.59 | 1515.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 1524.00 | 1493.28 | 1515.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1516.60 | 1497.16 | 1515.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 1518.60 | 1497.16 | 1515.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1519.50 | 1497.38 | 1515.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 1519.50 | 1497.38 | 1515.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1534.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1534.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1544.00 | 1499.08 | 1515.74 | SL hit (close>static) qty=1.00 sl=1534.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1490.30 | 1483.09 | 1500.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 1488.40 | 1486.38 | 1500.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 1486.50 | 1486.40 | 1500.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:30:00 | 1488.30 | 1486.72 | 1500.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 1488.00 | 1486.84 | 1500.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1532.90 | 1480.90 | 1495.05 | SL hit (close>static) qty=1.00 sl=1509.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1532.90 | 1480.90 | 1495.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1493.20 | 1488.34 | 1497.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 1491.50 | 1488.37 | 1497.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 1492.30 | 1488.51 | 1497.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1501.80 | 1488.64 | 1497.51 | SL hit (close>static) qty=1.00 sl=1499.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1501.80 | 1488.64 | 1497.51 | SL hit (close>static) qty=1.00 sl=1499.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 1491.20 | 1490.34 | 1497.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 1492.20 | 1490.52 | 1497.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1503.60 | 1486.13 | 1494.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1503.60 | 1486.13 | 1494.30 | SL hit (close>static) qty=1.00 sl=1499.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1503.60 | 1486.13 | 1494.30 | SL hit (close>static) qty=1.00 sl=1499.10 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.30 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.56 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.47 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.90 | EMA200 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 12:45:00 | 1582.50 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-05-15 09:15:00 | 1584.50 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-05-15 11:00:00 | 1575.00 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-16 11:15:00 | 1581.80 | 2025-05-29 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-16 14:15:00 | 1586.80 | 2025-05-29 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-05-19 09:15:00 | 1571.10 | 2025-06-10 10:15:00 | 1593.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-29 10:30:00 | 1585.10 | 2025-06-10 10:15:00 | 1593.80 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1584.40 | 2025-06-24 09:15:00 | 1598.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-24 14:45:00 | 1581.50 | 2025-06-25 09:15:00 | 1596.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-24 15:15:00 | 1579.60 | 2025-06-25 09:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-05 10:15:00 | 1437.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-09 10:15:00 | 1496.70 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-09-10 09:45:00 | 1522.80 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1516.80 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-12 09:30:00 | 1524.00 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1488.40 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-10-13 13:15:00 | 1486.50 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-14 12:30:00 | 1488.30 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-14 15:15:00 | 1488.00 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-10-28 12:00:00 | 1491.50 | 2025-10-28 14:15:00 | 1501.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1492.30 | 2025-10-28 14:15:00 | 1501.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-30 09:45:00 | 1491.20 | 2025-11-10 09:15:00 | 1503.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-30 13:15:00 | 1492.20 | 2025-11-10 09:15:00 | 1503.60 | STOP_HIT | 1.00 | -0.76% |
