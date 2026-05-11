# Tech Mahindra Ltd. (TECHM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (2875 bars)
- **Last close:** 1460.90
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
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** -1.16% / -1.55%
- **Sum % (uncompounded):** -13.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.15% | -4.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.15% | -4.3% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.96% | -9.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.96% | -9.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 2 | 16.7% | 0 | 11 | 1 | -1.16% | -13.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 1612.70 | 1477.56 | 1477.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 1626.90 | 1479.05 | 1477.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.90 | 1541.28 | 1516.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:00:00 | 1536.90 | 1541.28 | 1516.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.70 | 1639.94 | 1601.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1599.70 | 1639.94 | 1601.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1596.40 | 1639.50 | 1601.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1594.20 | 1639.50 | 1601.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1602.40 | 1637.41 | 1601.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1592.30 | 1637.41 | 1601.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1598.60 | 1637.02 | 1601.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1602.30 | 1634.74 | 1600.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 1570.20 | 1632.75 | 1600.58 | SL hit (close<static) qty=1.00 sl=1577.80 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 1451.50 | 1580.25 | 1580.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 1441.10 | 1576.29 | 1578.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.52 | 1543.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 1525.00 | 1520.52 | 1543.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.94 | 1535.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 1532.00 | 1514.94 | 1535.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1528.50 | 1505.39 | 1523.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1528.50 | 1505.39 | 1523.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1529.90 | 1505.63 | 1523.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 1535.00 | 1505.63 | 1523.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1526.50 | 1506.70 | 1524.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1514.70 | 1506.70 | 1524.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1521.60 | 1507.44 | 1523.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 1510.80 | 1508.47 | 1523.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 1533.00 | 1509.83 | 1523.51 | SL hit (close>static) qty=1.00 sl=1530.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1546.10 | 1472.93 | 1472.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.65 | 1474.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.91 | 1546.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:30:00 | 1579.90 | 1579.91 | 1546.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.66 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 1439.60 | 1588.09 | 1589.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.16 | 1473.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 1425.20 | 1416.16 | 1473.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.81 | 1461.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1455.80 | 1413.81 | 1461.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1451.40 | 1414.19 | 1461.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1437.50 | 1422.32 | 1461.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1440.80 | 1424.27 | 1461.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1462.10 | 1425.95 | 1459.40 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-11 14:45:00 | 1602.30 | 2025-07-14 11:15:00 | 1570.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-07-16 14:00:00 | 1603.30 | 2025-07-17 12:15:00 | 1566.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-09-15 10:00:00 | 1510.80 | 2025-09-16 14:15:00 | 1533.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1493.70 | 2025-09-26 11:15:00 | 1419.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1493.70 | 2025-10-09 12:15:00 | 1473.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-27 11:45:00 | 1510.20 | 2025-12-02 14:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-27 12:15:00 | 1512.20 | 2025-12-02 14:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1437.50 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-10 09:15:00 | 1440.80 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1437.50 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-22 14:15:00 | 1446.90 | 2026-04-22 15:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-30 13:15:00 | 1480.00 | STOP_HIT | 1.00 | -4.45% |
