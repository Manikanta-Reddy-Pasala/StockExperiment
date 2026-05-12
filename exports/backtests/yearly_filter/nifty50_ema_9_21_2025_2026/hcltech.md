# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 50 |
| ALERT2 | 47 |
| ALERT2_SKIP | 25 |
| ALERT3 | 117 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 52 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 38
- **Target hits / Stop hits / Partials:** 1 / 56 / 4
- **Avg / median % per leg:** 0.58% / -0.42%
- **Sum % (uncompounded):** 35.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 13 | 46.4% | 0 | 28 | 0 | -0.07% | -1.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.37% | -1.5% |
| BUY @ 3rd Alert (retest2) | 24 | 11 | 45.8% | 0 | 24 | 0 | -0.01% | -0.3% |
| SELL (all) | 33 | 10 | 30.3% | 1 | 28 | 4 | 1.13% | 37.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.25% | -1.3% |
| SELL @ 3rd Alert (retest2) | 32 | 10 | 31.2% | 1 | 27 | 4 | 1.20% | 38.5% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.55% | -2.8% |
| retest2 (combined) | 56 | 21 | 37.5% | 1 | 51 | 4 | 0.68% | 38.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1627.50 | 1583.02 | 1577.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1643.60 | 1601.88 | 1587.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1639.00 | 1640.15 | 1617.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 1636.50 | 1640.15 | 1617.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1616.10 | 1630.81 | 1619.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1616.10 | 1630.81 | 1619.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1620.60 | 1628.77 | 1620.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1633.40 | 1628.77 | 1620.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 1644.60 | 1652.68 | 1652.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 1644.60 | 1652.68 | 1652.96 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1656.10 | 1653.42 | 1653.14 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1648.30 | 1652.39 | 1652.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1644.00 | 1650.71 | 1651.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1647.00 | 1645.51 | 1648.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1637.80 | 1643.97 | 1647.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1621.20 | 1647.99 | 1648.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 15:00:00 | 1632.10 | 1636.03 | 1641.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1666.70 | 1642.80 | 1643.30 | SL hit (close>static) qty=1.00 sl=1650.80 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1668.30 | 1647.90 | 1645.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1673.60 | 1654.50 | 1650.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1650.80 | 1656.48 | 1657.15 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1664.70 | 1657.64 | 1656.96 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1654.50 | 1657.80 | 1658.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1635.70 | 1653.38 | 1656.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 1637.40 | 1632.69 | 1639.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1631.90 | 1632.53 | 1638.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 1635.40 | 1632.53 | 1638.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1632.10 | 1632.19 | 1636.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 1625.60 | 1633.42 | 1635.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 1650.90 | 1636.66 | 1636.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1650.90 | 1636.66 | 1636.42 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1628.20 | 1635.09 | 1635.75 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1638.00 | 1634.79 | 1634.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1653.80 | 1638.59 | 1636.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1696.70 | 1704.20 | 1689.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 1696.70 | 1704.20 | 1689.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1694.90 | 1701.94 | 1692.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1703.00 | 1701.94 | 1692.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 1697.40 | 1700.17 | 1695.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 1704.20 | 1717.45 | 1717.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1704.20 | 1717.45 | 1717.87 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1735.60 | 1718.39 | 1716.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1740.40 | 1727.38 | 1721.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1709.10 | 1725.25 | 1721.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1708.10 | 1721.82 | 1720.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1706.60 | 1721.82 | 1720.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 1713.40 | 1718.16 | 1718.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 1704.00 | 1714.25 | 1716.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 1710.80 | 1715.60 | 1716.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1710.60 | 1705.07 | 1710.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 1710.10 | 1706.66 | 1710.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 1710.60 | 1707.45 | 1710.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1719.20 | 1709.80 | 1711.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 1719.20 | 1709.80 | 1711.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1718.20 | 1711.48 | 1711.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:30:00 | 1717.90 | 1711.48 | 1711.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-25 14:15:00 | 1717.90 | 1712.76 | 1712.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1717.90 | 1712.76 | 1712.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 1721.40 | 1715.36 | 1713.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1722.50 | 1725.89 | 1721.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:15:00 | 1721.00 | 1725.89 | 1721.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1716.70 | 1724.05 | 1720.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 1716.70 | 1724.05 | 1720.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1724.80 | 1724.20 | 1721.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:30:00 | 1711.80 | 1724.20 | 1721.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1715.80 | 1723.24 | 1721.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1714.00 | 1723.24 | 1721.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1719.20 | 1722.43 | 1721.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1717.60 | 1722.43 | 1721.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1724.40 | 1722.83 | 1721.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 1725.90 | 1722.70 | 1721.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 1715.90 | 1724.29 | 1723.55 | SL hit (close<static) qty=1.00 sl=1720.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 1717.80 | 1722.99 | 1723.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 1712.70 | 1720.24 | 1721.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 1721.10 | 1720.41 | 1721.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1719.50 | 1720.23 | 1721.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 1712.70 | 1720.23 | 1721.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 1715.90 | 1719.69 | 1721.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 15:15:00 | 1715.00 | 1719.51 | 1720.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1722.10 | 1719.31 | 1720.46 | SL hit (close>static) qty=1.00 sl=1722.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1730.00 | 1721.44 | 1721.33 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 1718.70 | 1720.90 | 1721.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1714.50 | 1719.38 | 1720.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 1718.80 | 1715.88 | 1718.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1719.30 | 1716.56 | 1718.16 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1725.90 | 1719.90 | 1719.41 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1711.40 | 1719.16 | 1719.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1691.70 | 1708.38 | 1713.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1714.50 | 1707.03 | 1711.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1711.00 | 1707.82 | 1711.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1717.30 | 1707.82 | 1711.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1709.80 | 1708.22 | 1711.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1709.80 | 1708.22 | 1711.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1700.10 | 1706.60 | 1710.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1691.60 | 1706.60 | 1710.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1683.00 | 1701.88 | 1707.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 1680.30 | 1701.88 | 1707.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 1680.20 | 1697.54 | 1705.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 1679.40 | 1694.83 | 1703.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1596.28 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1596.19 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1595.43 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 1549.20 | 1546.84 | 1557.90 | SL hit (close>ema200) qty=0.50 sl=1546.84 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 1476.30 | 1466.89 | 1465.78 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 1463.30 | 1469.27 | 1469.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 1459.90 | 1466.55 | 1468.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1465.60 | 1463.09 | 1465.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1475.40 | 1465.55 | 1466.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1475.40 | 1465.55 | 1466.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1479.10 | 1468.26 | 1467.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 1484.30 | 1475.06 | 1471.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 1495.90 | 1497.62 | 1491.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1499.30 | 1502.32 | 1497.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 1499.30 | 1502.32 | 1497.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1494.70 | 1500.80 | 1497.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 1494.70 | 1500.80 | 1497.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1489.20 | 1498.48 | 1496.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 1489.20 | 1498.48 | 1496.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1489.00 | 1496.58 | 1495.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 1481.90 | 1496.58 | 1495.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1485.30 | 1494.32 | 1495.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1474.20 | 1483.39 | 1488.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 1479.70 | 1479.18 | 1483.51 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1466.60 | 1479.18 | 1483.51 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1479.10 | 1477.46 | 1481.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 1477.60 | 1477.46 | 1481.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 1485.00 | 1478.96 | 1482.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 1485.00 | 1478.96 | 1482.15 | SL hit (close>ema400) qty=1.00 sl=1482.15 alert=retest1 |

### Cycle 25 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1497.40 | 1485.76 | 1484.73 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1472.90 | 1486.07 | 1487.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1466.00 | 1476.13 | 1481.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 1499.10 | 1477.82 | 1481.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1499.00 | 1482.05 | 1482.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1499.00 | 1482.05 | 1482.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1505.90 | 1486.82 | 1485.04 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1453.00 | 1486.59 | 1489.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1451.70 | 1471.97 | 1481.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1461.10 | 1459.81 | 1470.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1461.10 | 1459.81 | 1470.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1461.50 | 1457.66 | 1464.76 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1470.00 | 1466.69 | 1466.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 1473.40 | 1468.03 | 1467.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1462.60 | 1467.90 | 1467.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1465.20 | 1467.36 | 1467.16 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1465.00 | 1466.89 | 1466.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 09:15:00 | 1462.00 | 1465.91 | 1466.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1468.10 | 1463.00 | 1464.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1465.20 | 1463.44 | 1464.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1460.30 | 1463.44 | 1464.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1460.60 | 1430.92 | 1428.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1460.60 | 1430.92 | 1428.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1468.40 | 1438.41 | 1432.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 1464.60 | 1465.16 | 1458.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1464.60 | 1465.16 | 1458.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1455.70 | 1464.08 | 1460.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1455.70 | 1464.08 | 1460.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1454.10 | 1462.09 | 1459.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 1454.10 | 1462.09 | 1459.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1464.90 | 1462.36 | 1460.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1462.80 | 1462.36 | 1460.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1472.10 | 1465.50 | 1462.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 1473.30 | 1467.86 | 1464.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 1474.70 | 1469.23 | 1465.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 1470.30 | 1481.05 | 1481.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1470.30 | 1481.05 | 1481.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1465.90 | 1476.70 | 1479.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1438.00 | 1434.82 | 1444.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 1438.00 | 1434.82 | 1444.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1441.10 | 1436.47 | 1443.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1442.40 | 1436.47 | 1443.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1441.30 | 1437.44 | 1443.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 1439.60 | 1438.17 | 1443.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1431.60 | 1439.32 | 1443.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1410.50 | 1392.78 | 1392.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1410.50 | 1392.78 | 1392.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 1414.60 | 1397.14 | 1394.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1487.50 | 1488.00 | 1475.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 14:15:00 | 1496.50 | 1488.27 | 1479.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 15:15:00 | 1494.10 | 1489.18 | 1480.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1492.00 | 1498.18 | 1491.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1504.50 | 1498.18 | 1491.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1497.50 | 1498.04 | 1492.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1497.70 | 1496.96 | 1492.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 1498.40 | 1496.64 | 1493.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1494.90 | 1496.29 | 1493.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 1494.60 | 1496.29 | 1493.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1497.60 | 1507.56 | 1502.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1497.60 | 1507.56 | 1502.56 | SL hit (close<ema400) qty=1.00 sl=1502.56 alert=retest1 |

### Cycle 34 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1489.30 | 1499.31 | 1499.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 1484.40 | 1496.33 | 1498.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 1503.40 | 1495.10 | 1497.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1494.90 | 1495.06 | 1497.16 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1507.00 | 1498.54 | 1498.44 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1494.90 | 1498.11 | 1498.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 1486.50 | 1495.66 | 1497.10 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1533.20 | 1503.17 | 1500.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1539.50 | 1510.43 | 1503.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1522.20 | 1524.00 | 1513.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1522.20 | 1524.00 | 1513.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1519.90 | 1523.45 | 1517.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 1518.30 | 1523.45 | 1517.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1523.00 | 1523.36 | 1518.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 1516.90 | 1523.36 | 1518.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1530.30 | 1533.71 | 1528.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1530.30 | 1533.71 | 1528.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1519.20 | 1530.81 | 1527.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 1519.20 | 1530.81 | 1527.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1514.10 | 1527.47 | 1526.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1514.10 | 1527.47 | 1526.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 1515.50 | 1524.10 | 1525.07 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1533.40 | 1525.47 | 1525.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1548.00 | 1529.98 | 1527.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1545.00 | 1546.41 | 1538.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1545.00 | 1546.41 | 1538.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1546.40 | 1546.41 | 1539.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1541.70 | 1546.41 | 1539.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1548.10 | 1546.77 | 1541.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1541.60 | 1546.77 | 1541.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1551.60 | 1548.57 | 1543.43 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 1532.90 | 1541.85 | 1542.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 1527.40 | 1538.27 | 1540.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1530.00 | 1528.42 | 1532.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 1530.00 | 1528.42 | 1532.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1525.20 | 1527.46 | 1531.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:45:00 | 1530.30 | 1527.46 | 1531.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1532.00 | 1517.08 | 1521.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1532.00 | 1517.08 | 1521.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1541.80 | 1522.02 | 1523.17 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1537.00 | 1525.02 | 1524.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 1549.40 | 1536.51 | 1530.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1588.20 | 1594.02 | 1583.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1596.30 | 1594.48 | 1584.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1576.50 | 1590.50 | 1584.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1576.50 | 1590.50 | 1584.21 | SL hit (close<ema400) qty=1.00 sl=1584.21 alert=retest1 |

### Cycle 42 — SELL (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 13:15:00 | 1613.20 | 1631.55 | 1632.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1607.50 | 1626.74 | 1630.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 1630.40 | 1623.80 | 1628.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1625.90 | 1624.22 | 1627.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 1628.50 | 1624.22 | 1627.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1625.10 | 1624.40 | 1627.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 1630.20 | 1624.40 | 1627.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1623.00 | 1624.12 | 1627.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 1622.00 | 1623.69 | 1626.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1623.20 | 1616.53 | 1616.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1623.20 | 1616.53 | 1616.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 1629.60 | 1622.82 | 1620.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 1622.80 | 1627.57 | 1624.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1623.80 | 1626.82 | 1624.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1633.90 | 1626.82 | 1624.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:45:00 | 1632.00 | 1634.61 | 1631.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1658.20 | 1671.30 | 1671.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 1658.20 | 1671.30 | 1671.91 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 1672.40 | 1667.81 | 1667.53 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 1662.50 | 1666.75 | 1667.07 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 1671.20 | 1667.61 | 1667.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1672.90 | 1669.21 | 1668.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1648.40 | 1668.27 | 1670.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 1641.20 | 1655.92 | 1658.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1662.50 | 1648.34 | 1651.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1665.50 | 1651.77 | 1653.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1665.50 | 1651.77 | 1653.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1665.00 | 1654.42 | 1654.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 1670.70 | 1657.67 | 1655.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1674.30 | 1674.75 | 1668.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 1674.30 | 1674.75 | 1668.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1667.00 | 1673.32 | 1670.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1667.10 | 1673.32 | 1670.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1660.70 | 1670.80 | 1669.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1660.70 | 1670.80 | 1669.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1657.30 | 1668.10 | 1668.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1653.20 | 1661.69 | 1665.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1629.30 | 1625.35 | 1633.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1629.30 | 1625.35 | 1633.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1626.70 | 1625.04 | 1630.94 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 1642.80 | 1634.94 | 1634.32 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 1631.10 | 1633.96 | 1633.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1630.60 | 1633.29 | 1633.66 | Break + close below crossover candle low |

### Cycle 53 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1640.50 | 1634.73 | 1634.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1641.70 | 1637.49 | 1635.85 | Break + close above crossover candle high |

### Cycle 54 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1599.70 | 1630.33 | 1632.90 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1642.00 | 1621.24 | 1621.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 11:15:00 | 1644.30 | 1628.90 | 1624.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 1641.10 | 1646.89 | 1640.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1659.60 | 1646.89 | 1640.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1650.20 | 1658.45 | 1651.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1650.20 | 1658.45 | 1651.97 | SL hit (close<ema400) qty=1.00 sl=1651.97 alert=retest1 |

### Cycle 56 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 1678.60 | 1689.35 | 1690.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1698.50 | 1687.38 | 1687.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1704.00 | 1693.40 | 1690.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1701.20 | 1704.38 | 1698.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 1701.20 | 1704.38 | 1698.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1706.70 | 1704.84 | 1698.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:30:00 | 1716.30 | 1707.28 | 1701.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 1717.00 | 1708.25 | 1702.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:45:00 | 1714.70 | 1709.06 | 1703.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1720.00 | 1714.21 | 1708.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1703.10 | 1716.64 | 1712.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1703.10 | 1716.64 | 1712.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1710.00 | 1715.31 | 1712.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 1701.10 | 1715.31 | 1712.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1694.30 | 1713.28 | 1713.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1694.30 | 1713.28 | 1713.08 | SL hit (close<static) qty=1.00 sl=1698.80 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 1697.00 | 1710.03 | 1711.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 1686.00 | 1696.32 | 1702.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 1684.30 | 1681.03 | 1689.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:45:00 | 1686.40 | 1681.03 | 1689.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1710.90 | 1685.33 | 1688.37 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1718.00 | 1695.89 | 1692.89 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 1610.70 | 1680.02 | 1687.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1579.30 | 1591.15 | 1599.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1462.00 | 1458.11 | 1477.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1462.00 | 1458.11 | 1477.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1489.20 | 1465.52 | 1477.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1489.20 | 1465.52 | 1477.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1489.30 | 1470.28 | 1478.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1486.20 | 1470.28 | 1478.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1483.50 | 1482.86 | 1482.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1474.00 | 1482.86 | 1482.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1490.80 | 1472.50 | 1474.92 | SL hit (close>static) qty=1.00 sl=1484.80 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 1391.20 | 1386.64 | 1386.11 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1383.60 | 1385.78 | 1385.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 09:15:00 | 1346.80 | 1364.30 | 1371.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1351.80 | 1350.49 | 1360.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1356.90 | 1350.49 | 1360.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1369.00 | 1355.03 | 1360.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1341.00 | 1359.55 | 1361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 13:00:00 | 1353.00 | 1357.75 | 1359.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 13:45:00 | 1352.70 | 1356.76 | 1359.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1352.60 | 1356.93 | 1358.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1362.20 | 1357.98 | 1359.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1351.50 | 1357.98 | 1359.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1352.30 | 1356.84 | 1358.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1363.00 | 1358.11 | 1358.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 1363.00 | 1358.11 | 1358.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 1356.80 | 1357.84 | 1357.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1348.90 | 1355.68 | 1356.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 1361.30 | 1355.13 | 1356.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1360.10 | 1356.13 | 1356.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 1362.50 | 1356.13 | 1356.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1357.00 | 1356.30 | 1356.61 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 1358.60 | 1357.05 | 1356.92 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1343.30 | 1354.29 | 1355.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1332.80 | 1348.24 | 1352.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1333.00 | 1323.05 | 1333.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1366.60 | 1331.09 | 1331.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1373.80 | 1339.63 | 1334.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1336.80 | 1351.53 | 1345.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1326.90 | 1346.61 | 1343.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1326.90 | 1346.61 | 1343.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1324.00 | 1339.33 | 1340.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1322.80 | 1336.03 | 1338.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1346.40 | 1331.94 | 1335.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1349.00 | 1335.35 | 1336.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1351.50 | 1335.35 | 1336.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 1340.50 | 1338.00 | 1337.93 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 1330.40 | 1337.67 | 1337.86 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 1339.90 | 1338.11 | 1338.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-23 10:15:00 | 1349.80 | 1341.71 | 1339.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1380.90 | 1387.24 | 1375.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 1380.90 | 1387.24 | 1375.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1389.70 | 1387.47 | 1377.60 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1360.10 | 1374.79 | 1375.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1346.60 | 1369.15 | 1372.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 1363.00 | 1364.33 | 1365.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 1362.80 | 1359.67 | 1361.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 1378.20 | 1363.38 | 1363.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1378.20 | 1363.38 | 1363.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1399.40 | 1370.58 | 1366.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1439.40 | 1446.82 | 1432.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 1439.20 | 1446.82 | 1432.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1433.70 | 1451.48 | 1442.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 1439.30 | 1451.48 | 1442.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1429.30 | 1447.04 | 1441.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 1429.30 | 1447.04 | 1441.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1450.80 | 1444.16 | 1441.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 1443.40 | 1444.16 | 1441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1427.70 | 1442.12 | 1440.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1426.00 | 1442.12 | 1440.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1423.40 | 1438.38 | 1439.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 11:15:00 | 1422.90 | 1435.28 | 1437.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1448.70 | 1438.30 | 1437.59 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 1433.20 | 1440.55 | 1441.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 1429.60 | 1434.93 | 1437.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 1437.00 | 1433.66 | 1436.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1442.30 | 1435.39 | 1436.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 1442.30 | 1435.39 | 1436.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 1449.40 | 1438.19 | 1438.03 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 1436.60 | 1437.87 | 1437.90 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1442.00 | 1438.70 | 1438.27 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1305.20 | 1412.41 | 1426.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 1299.80 | 1389.88 | 1414.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 1220.30 | 1220.00 | 1254.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 1220.30 | 1220.00 | 1254.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1207.10 | 1201.48 | 1209.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 1206.20 | 1201.48 | 1209.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1210.00 | 1203.18 | 1209.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 1210.00 | 1203.18 | 1209.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1198.10 | 1202.17 | 1208.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1209.50 | 1202.17 | 1208.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1196.70 | 1200.95 | 1206.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 1193.30 | 1199.24 | 1204.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 1194.40 | 1199.11 | 1201.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1193.50 | 1190.38 | 1192.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 1199.40 | 1193.15 | 1193.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 1199.40 | 1193.15 | 1193.06 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1633.40 | 2025-05-19 13:15:00 | 1644.60 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1621.20 | 2025-05-23 09:15:00 | 1666.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-05-22 15:00:00 | 1632.10 | 2025-05-23 09:15:00 | 1666.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-04 15:00:00 | 1625.60 | 2025-06-05 11:15:00 | 1650.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1703.00 | 2025-06-19 09:15:00 | 1704.20 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-06-13 15:15:00 | 1697.40 | 2025-06-19 09:15:00 | 1704.20 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-06-24 13:15:00 | 1710.80 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1710.60 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-06-25 10:45:00 | 1710.10 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-25 12:00:00 | 1710.60 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-30 13:15:00 | 1725.90 | 2025-07-01 12:15:00 | 1715.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-02 13:15:00 | 1712.70 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-02 14:15:00 | 1715.90 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-02 15:15:00 | 1715.00 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-09 10:15:00 | 1680.30 | 2025-07-15 09:15:00 | 1596.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 11:00:00 | 1680.20 | 2025-07-15 09:15:00 | 1596.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 12:15:00 | 1679.40 | 2025-07-15 09:15:00 | 1595.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 10:15:00 | 1680.30 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2025-07-09 11:00:00 | 1680.20 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2025-07-09 12:15:00 | 1679.40 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.75% |
| SELL | retest1 | 2025-08-20 09:15:00 | 1466.60 | 2025-08-20 11:15:00 | 1485.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1460.30 | 2025-09-10 09:15:00 | 1460.60 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-09-16 12:30:00 | 1473.30 | 2025-09-19 12:15:00 | 1470.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-09-16 14:00:00 | 1474.70 | 2025-09-19 12:15:00 | 1470.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-09-24 14:15:00 | 1439.60 | 2025-10-06 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1431.60 | 2025-10-06 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest1 | 2025-10-13 14:15:00 | 1496.50 | 2025-10-17 09:15:00 | 1497.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest1 | 2025-10-13 15:15:00 | 1494.10 | 2025-10-17 09:15:00 | 1497.60 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1504.50 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1497.50 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-15 12:15:00 | 1497.70 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-15 14:15:00 | 1498.40 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-11-14 11:00:00 | 1596.30 | 2025-11-14 12:15:00 | 1576.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-14 14:30:00 | 1585.10 | 2025-11-21 13:15:00 | 1613.20 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2025-11-24 14:00:00 | 1622.00 | 2025-11-26 13:15:00 | 1623.20 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1633.90 | 2025-12-09 14:15:00 | 1658.20 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-12-02 11:45:00 | 1632.00 | 2025-12-09 14:15:00 | 1658.20 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest1 | 2026-01-09 09:15:00 | 1659.60 | 2026-01-12 09:15:00 | 1650.20 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-12 12:00:00 | 1655.00 | 2026-01-13 09:15:00 | 1643.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-12 13:30:00 | 1655.70 | 2026-01-13 09:15:00 | 1643.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-13 10:30:00 | 1656.80 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2026-01-13 14:15:00 | 1658.60 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2026-01-14 11:15:00 | 1669.70 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-01-14 14:30:00 | 1670.00 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2026-01-14 15:00:00 | 1668.30 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2026-01-27 09:30:00 | 1716.30 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-27 11:30:00 | 1717.00 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-27 12:45:00 | 1714.70 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-28 09:30:00 | 1720.00 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-02-18 09:15:00 | 1474.00 | 2026-02-19 09:15:00 | 1490.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1480.00 | 2026-02-24 09:15:00 | 1406.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1480.00 | 2026-02-24 12:15:00 | 1332.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1341.00 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-09 13:00:00 | 1353.00 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-09 13:45:00 | 1352.70 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-09 14:30:00 | 1352.60 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-01 13:00:00 | 1363.00 | 2026-04-02 12:15:00 | 1378.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-04-02 12:00:00 | 1362.80 | 2026-04-02 12:15:00 | 1378.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-05-04 13:15:00 | 1193.30 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-05-06 12:00:00 | 1194.40 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-08 11:45:00 | 1193.50 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.49% |
