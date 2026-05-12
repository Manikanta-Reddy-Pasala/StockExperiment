# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 1 / 14 / 1
- **Avg / median % per leg:** -0.35% / -1.44%
- **Sum % (uncompounded):** -5.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 1 | 10 | 0 | -0.43% | -4.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 1 | 10 | 0 | -0.43% | -4.7% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.17% | -0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.17% | -0.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 4 | 25.0% | 1 | 14 | 1 | -0.35% | -5.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1550.00 | 1561.90 | 1561.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1583.10 | 1562.02 | 1561.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 1593.50 | 1562.76 | 1562.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1672.20 | 1673.54 | 1641.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:45:00 | 1674.80 | 1673.54 | 1641.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1650.80 | 1675.26 | 1650.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1650.80 | 1675.26 | 1650.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1639.90 | 1674.90 | 1650.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1653.70 | 1674.90 | 1650.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1653.00 | 1674.67 | 1650.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 1629.90 | 1673.84 | 1650.14 | SL hit (close<static) qty=1.00 sl=1633.30 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 1597.00 | 1649.33 | 1649.41 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 1708.40 | 1649.17 | 1649.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1721.00 | 1657.05 | 1653.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1753.20 | 1771.98 | 1726.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 1753.20 | 1771.98 | 1726.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1729.20 | 1770.76 | 1726.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 1731.40 | 1770.76 | 1726.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1720.00 | 1769.17 | 1727.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 1720.00 | 1769.17 | 1727.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1729.10 | 1768.77 | 1727.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 13:15:00 | 1734.30 | 1768.77 | 1727.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:30:00 | 1734.30 | 1768.14 | 1727.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 1714.90 | 1766.93 | 1727.27 | SL hit (close<static) qty=1.00 sl=1719.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1560.30 | 1704.12 | 1704.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 1550.00 | 1701.31 | 1702.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1627.50 | 1618.76 | 1653.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 11:15:00 | 1654.10 | 1619.37 | 1653.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1654.10 | 1619.37 | 1653.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 1654.10 | 1619.37 | 1653.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 1649.80 | 1619.67 | 1653.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:30:00 | 1650.10 | 1619.67 | 1653.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1661.40 | 1620.09 | 1653.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 1661.40 | 1620.09 | 1653.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1653.50 | 1620.42 | 1653.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 1650.60 | 1620.42 | 1653.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1699.90 | 1622.95 | 1653.71 | SL hit (close>static) qty=1.00 sl=1662.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-07 15:15:00 | 1510.00 | 2025-08-18 09:15:00 | 1661.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1513.80 | 2025-11-03 10:15:00 | 1550.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1653.70 | 2025-12-30 11:15:00 | 1629.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1653.00 | 2025-12-30 11:15:00 | 1629.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1654.80 | 2026-01-14 09:15:00 | 1630.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-13 10:45:00 | 1655.10 | 2026-01-14 09:15:00 | 1630.90 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-05 13:15:00 | 1734.30 | 2026-03-06 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-05 14:30:00 | 1734.30 | 2026-03-06 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-10 13:30:00 | 1736.80 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-03-10 14:45:00 | 1730.40 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-03-11 10:15:00 | 1749.90 | 2026-03-12 09:15:00 | 1686.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-10 15:15:00 | 1650.60 | 2026-04-15 09:15:00 | 1699.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-04-21 13:15:00 | 1651.70 | 2026-04-24 10:15:00 | 1569.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 13:15:00 | 1651.70 | 2026-05-06 09:15:00 | 1635.80 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-04-21 14:00:00 | 1648.00 | 2026-05-07 10:15:00 | 1679.60 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1648.50 | 2026-05-07 10:15:00 | 1679.60 | STOP_HIT | 1.00 | -1.89% |
