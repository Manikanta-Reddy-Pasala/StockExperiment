# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 54 |
| ALERT2 | 55 |
| ALERT2_SKIP | 30 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 67 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 23 / 46
- **Target hits / Stop hits / Partials:** 0 / 65 / 4
- **Avg / median % per leg:** -0.01% / -0.51%
- **Sum % (uncompounded):** -0.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 10 | 26.3% | 0 | 38 | 0 | -0.16% | -6.1% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.12% | 0.2% |
| BUY @ 3rd Alert (retest2) | 36 | 8 | 22.2% | 0 | 36 | 0 | -0.18% | -6.3% |
| SELL (all) | 31 | 13 | 41.9% | 0 | 27 | 4 | 0.17% | 5.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 13 | 41.9% | 0 | 27 | 4 | 0.17% | 5.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.12% | 0.2% |
| retest2 (combined) | 67 | 21 | 31.3% | 0 | 63 | 4 | -0.01% | -0.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1737.70 | 1711.86 | 1710.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 1740.00 | 1717.48 | 1713.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1735.10 | 1736.77 | 1730.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 1735.10 | 1736.77 | 1730.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1729.10 | 1735.23 | 1729.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 1729.10 | 1735.23 | 1729.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1729.00 | 1733.99 | 1729.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 1740.80 | 1733.99 | 1729.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 1722.60 | 1731.43 | 1729.99 | SL hit (close<static) qty=1.00 sl=1726.60 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1710.40 | 1727.22 | 1728.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1708.40 | 1723.46 | 1726.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1745.20 | 1725.01 | 1726.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1740.70 | 1728.15 | 1727.80 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1712.50 | 1727.67 | 1728.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 1670.90 | 1711.65 | 1719.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 15:15:00 | 1689.00 | 1687.16 | 1701.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:15:00 | 1688.20 | 1687.16 | 1701.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1688.30 | 1687.39 | 1699.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1681.10 | 1686.13 | 1698.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 1681.30 | 1679.57 | 1686.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 1679.60 | 1679.57 | 1686.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1681.50 | 1680.31 | 1685.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1681.50 | 1680.55 | 1685.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1683.10 | 1680.55 | 1685.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1681.40 | 1680.72 | 1684.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 1667.60 | 1677.28 | 1681.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:45:00 | 1666.70 | 1674.83 | 1680.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1687.00 | 1678.23 | 1680.01 | SL hit (close>static) qty=1.00 sl=1686.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1687.00 | 1678.23 | 1680.01 | SL hit (close>static) qty=1.00 sl=1686.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 1699.50 | 1685.64 | 1683.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1684.80 | 1688.37 | 1685.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1685.00 | 1687.70 | 1685.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 1684.10 | 1687.70 | 1685.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1685.20 | 1687.20 | 1685.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 1685.70 | 1687.20 | 1685.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1686.70 | 1687.10 | 1685.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 1682.60 | 1687.10 | 1685.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1676.10 | 1684.90 | 1684.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1676.10 | 1684.90 | 1684.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1674.90 | 1682.90 | 1683.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 1661.50 | 1672.19 | 1676.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1670.60 | 1670.09 | 1674.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 1670.60 | 1670.09 | 1674.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1665.60 | 1662.94 | 1667.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1665.60 | 1662.94 | 1667.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1662.70 | 1662.89 | 1667.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1662.70 | 1662.89 | 1667.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1670.70 | 1664.63 | 1667.49 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1687.30 | 1671.00 | 1670.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 13:15:00 | 1692.00 | 1683.99 | 1679.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 1684.00 | 1688.05 | 1683.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1682.40 | 1686.92 | 1683.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 1682.00 | 1686.92 | 1683.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1681.40 | 1685.82 | 1683.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 1680.70 | 1685.82 | 1683.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1690.10 | 1686.67 | 1683.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1721.80 | 1688.57 | 1686.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1679.00 | 1691.12 | 1690.97 | SL hit (close<static) qty=1.00 sl=1680.80 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 1683.30 | 1689.56 | 1690.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1675.40 | 1686.27 | 1688.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1686.50 | 1683.60 | 1686.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1686.40 | 1684.16 | 1686.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1685.80 | 1684.16 | 1686.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1679.10 | 1683.14 | 1685.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1665.50 | 1683.14 | 1685.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 1660.90 | 1655.17 | 1655.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 1660.90 | 1655.17 | 1655.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1665.30 | 1657.19 | 1656.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1652.20 | 1657.16 | 1656.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1654.90 | 1656.70 | 1656.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 1660.40 | 1656.72 | 1656.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 1662.20 | 1656.82 | 1656.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 1653.40 | 1664.93 | 1665.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 1653.40 | 1664.93 | 1665.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1653.40 | 1664.93 | 1665.65 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 1669.30 | 1665.95 | 1665.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1674.20 | 1667.67 | 1666.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1684.80 | 1687.21 | 1679.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1684.80 | 1687.21 | 1679.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1686.90 | 1687.15 | 1680.28 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1665.10 | 1676.16 | 1676.77 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 1680.70 | 1673.92 | 1673.79 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1671.20 | 1673.38 | 1673.55 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 1677.60 | 1674.22 | 1673.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 1689.60 | 1678.22 | 1675.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 1680.00 | 1680.38 | 1677.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 1680.00 | 1680.38 | 1677.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1678.90 | 1680.69 | 1678.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1678.80 | 1680.69 | 1678.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1676.40 | 1679.84 | 1678.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1675.40 | 1679.84 | 1678.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1678.20 | 1679.51 | 1678.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 1672.90 | 1679.51 | 1678.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1682.70 | 1680.15 | 1678.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 1680.70 | 1680.15 | 1678.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1679.90 | 1680.10 | 1678.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1678.60 | 1680.10 | 1678.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1680.00 | 1680.08 | 1678.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1679.50 | 1680.08 | 1678.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1676.50 | 1679.36 | 1678.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 1676.00 | 1679.36 | 1678.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1675.90 | 1678.67 | 1678.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1675.90 | 1678.67 | 1678.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1675.90 | 1678.12 | 1678.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1670.90 | 1678.12 | 1678.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1667.00 | 1675.89 | 1677.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1661.30 | 1673.63 | 1675.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1668.00 | 1667.79 | 1671.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1668.00 | 1667.79 | 1671.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1673.10 | 1668.86 | 1671.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1673.10 | 1668.86 | 1671.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1670.80 | 1669.24 | 1671.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1674.80 | 1669.24 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1668.20 | 1669.04 | 1671.29 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 1673.70 | 1667.27 | 1667.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1674.60 | 1669.35 | 1668.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 1680.80 | 1681.91 | 1676.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1680.80 | 1681.91 | 1676.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1701.00 | 1709.05 | 1702.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 1701.00 | 1709.05 | 1702.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1701.90 | 1707.62 | 1702.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1714.20 | 1707.62 | 1702.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1705.50 | 1707.19 | 1702.95 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1688.50 | 1700.79 | 1702.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1674.10 | 1690.52 | 1694.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 1685.50 | 1680.82 | 1685.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1686.00 | 1681.86 | 1685.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 1681.60 | 1681.86 | 1685.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 1692.50 | 1687.04 | 1686.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 1692.50 | 1687.04 | 1686.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 10:15:00 | 1697.40 | 1691.26 | 1688.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 15:15:00 | 1699.00 | 1700.12 | 1696.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:15:00 | 1712.70 | 1700.12 | 1696.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1716.40 | 1725.29 | 1717.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1734.70 | 1726.43 | 1718.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 1732.90 | 1727.73 | 1720.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1735.00 | 1729.34 | 1722.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1702.80 | 1724.03 | 1720.37 | SL hit (close<static) qty=1.00 sl=1713.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1702.80 | 1724.03 | 1720.37 | SL hit (close<static) qty=1.00 sl=1713.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1702.80 | 1724.03 | 1720.37 | SL hit (close<static) qty=1.00 sl=1713.40 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1638.70 | 1703.12 | 1711.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 1604.20 | 1620.31 | 1633.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1599.80 | 1592.40 | 1608.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1599.80 | 1592.40 | 1608.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1603.90 | 1592.88 | 1598.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1603.90 | 1592.88 | 1598.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1599.90 | 1594.28 | 1598.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1596.40 | 1594.28 | 1598.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1611.60 | 1600.98 | 1600.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1611.60 | 1600.98 | 1600.46 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 1624.30 | 1629.38 | 1629.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 1618.30 | 1627.04 | 1628.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1626.20 | 1626.13 | 1627.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1632.80 | 1627.46 | 1627.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1632.80 | 1627.46 | 1627.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 1635.00 | 1628.97 | 1628.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1645.00 | 1633.05 | 1630.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 1642.70 | 1643.61 | 1639.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 1642.70 | 1643.61 | 1639.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1639.50 | 1642.79 | 1639.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 1635.30 | 1642.79 | 1639.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1641.60 | 1642.55 | 1639.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 1641.60 | 1642.55 | 1639.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1647.00 | 1643.44 | 1640.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1649.60 | 1645.34 | 1642.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1619.80 | 1643.71 | 1642.52 | SL hit (close<static) qty=1.00 sl=1639.90 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1619.10 | 1638.79 | 1640.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1610.10 | 1629.50 | 1635.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 1590.70 | 1587.48 | 1598.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 1590.70 | 1587.48 | 1598.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1595.20 | 1589.03 | 1598.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1590.10 | 1590.96 | 1597.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 1588.30 | 1590.43 | 1596.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1580.00 | 1574.63 | 1574.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1580.00 | 1574.63 | 1574.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 1580.00 | 1574.63 | 1574.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1586.90 | 1577.09 | 1575.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 1588.20 | 1590.05 | 1584.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 1588.20 | 1590.05 | 1584.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1580.20 | 1588.79 | 1585.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1580.20 | 1588.79 | 1585.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1580.00 | 1587.03 | 1585.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1589.30 | 1587.03 | 1585.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1582.30 | 1585.33 | 1584.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 1582.50 | 1585.33 | 1584.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 1582.50 | 1584.01 | 1584.07 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1584.60 | 1584.12 | 1584.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1591.30 | 1585.56 | 1584.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1587.00 | 1589.08 | 1586.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1593.00 | 1589.86 | 1587.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 1596.00 | 1590.99 | 1588.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 1630.50 | 1640.42 | 1640.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1630.50 | 1640.42 | 1640.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 1627.50 | 1632.98 | 1636.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1639.90 | 1632.08 | 1634.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1640.30 | 1633.72 | 1634.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 1640.30 | 1633.72 | 1634.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1609.40 | 1596.42 | 1608.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 1614.80 | 1596.42 | 1608.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1611.60 | 1599.46 | 1608.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1613.00 | 1599.46 | 1608.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1603.00 | 1599.04 | 1602.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1602.10 | 1599.04 | 1602.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1594.70 | 1598.18 | 1602.15 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 1630.70 | 1604.09 | 1603.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1657.00 | 1640.26 | 1629.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1642.80 | 1651.64 | 1644.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1645.10 | 1650.34 | 1644.62 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1630.90 | 1641.59 | 1641.89 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 1653.30 | 1643.28 | 1642.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 1658.80 | 1647.96 | 1644.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1663.50 | 1665.03 | 1657.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:45:00 | 1663.40 | 1665.03 | 1657.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1663.30 | 1663.74 | 1658.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 1665.60 | 1663.74 | 1658.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 1665.10 | 1663.85 | 1658.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1653.40 | 1659.63 | 1659.60 | SL hit (close<static) qty=1.00 sl=1655.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1653.40 | 1659.63 | 1659.60 | SL hit (close<static) qty=1.00 sl=1655.80 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 1654.70 | 1658.64 | 1659.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 1645.70 | 1653.81 | 1656.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1655.20 | 1652.01 | 1654.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1659.60 | 1653.53 | 1654.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 1661.40 | 1653.53 | 1654.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1656.40 | 1654.10 | 1655.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1664.50 | 1654.10 | 1655.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1670.80 | 1657.44 | 1656.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1674.30 | 1660.82 | 1658.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 1695.00 | 1695.49 | 1686.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 1695.00 | 1695.49 | 1686.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1686.70 | 1693.01 | 1687.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1686.70 | 1693.01 | 1687.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1695.00 | 1693.41 | 1687.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1689.60 | 1693.41 | 1687.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1692.40 | 1693.21 | 1688.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 1695.00 | 1693.21 | 1688.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1690.20 | 1692.61 | 1688.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1695.30 | 1693.61 | 1689.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:45:00 | 1695.70 | 1694.31 | 1690.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 1699.30 | 1695.31 | 1691.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1693.60 | 1694.80 | 1691.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1696.50 | 1695.14 | 1692.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 1700.20 | 1694.61 | 1693.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1687.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1687.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1687.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1687.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1691.50 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1681.00 | 1690.35 | 1691.30 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1699.50 | 1691.30 | 1691.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1712.90 | 1695.62 | 1693.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 1684.50 | 1703.71 | 1699.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1695.60 | 1702.09 | 1699.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1696.10 | 1702.09 | 1699.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 1695.80 | 1700.83 | 1699.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1697.00 | 1700.63 | 1699.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 1696.00 | 1699.52 | 1699.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1689.00 | 1695.66 | 1697.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1696.70 | 1695.12 | 1696.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1695.90 | 1695.28 | 1696.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1697.50 | 1695.28 | 1696.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1699.70 | 1696.16 | 1696.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 1699.70 | 1696.16 | 1696.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1695.50 | 1696.03 | 1696.82 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1705.00 | 1698.71 | 1697.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 15:15:00 | 1706.60 | 1700.29 | 1698.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1696.80 | 1699.59 | 1698.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1689.30 | 1697.53 | 1697.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1684.40 | 1693.07 | 1695.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1704.60 | 1694.10 | 1695.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1672.20 | 1689.72 | 1693.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 1668.30 | 1689.72 | 1693.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1696.40 | 1691.50 | 1691.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1696.40 | 1691.50 | 1691.35 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 1690.00 | 1691.20 | 1691.23 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1698.30 | 1692.62 | 1691.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1700.60 | 1694.22 | 1692.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1699.20 | 1694.84 | 1693.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 1698.40 | 1696.41 | 1694.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1797.10 | 1810.15 | 1810.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1797.10 | 1810.15 | 1810.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1797.10 | 1810.15 | 1810.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 1796.10 | 1807.34 | 1809.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1801.60 | 1800.53 | 1805.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 1801.60 | 1800.53 | 1805.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1794.30 | 1799.99 | 1804.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1785.30 | 1797.05 | 1802.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1806.00 | 1803.27 | 1803.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 1806.00 | 1803.27 | 1803.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1816.40 | 1806.49 | 1804.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1806.50 | 1809.50 | 1806.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1809.00 | 1809.40 | 1806.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1807.30 | 1809.40 | 1806.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1809.00 | 1809.32 | 1807.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1809.00 | 1809.32 | 1807.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1808.30 | 1809.11 | 1807.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1808.30 | 1809.11 | 1807.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1807.40 | 1808.77 | 1807.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1807.40 | 1808.77 | 1807.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1806.00 | 1808.22 | 1807.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1801.80 | 1808.22 | 1807.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1805.00 | 1807.57 | 1806.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1792.70 | 1807.57 | 1806.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1798.00 | 1805.66 | 1806.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1785.70 | 1796.23 | 1800.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 1795.00 | 1795.03 | 1798.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1785.20 | 1783.52 | 1788.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1790.20 | 1783.52 | 1788.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1788.50 | 1784.52 | 1788.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1786.70 | 1784.11 | 1787.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1795.60 | 1786.41 | 1788.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1795.60 | 1786.41 | 1788.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1798.20 | 1788.77 | 1789.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1798.20 | 1788.77 | 1789.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1798.00 | 1790.61 | 1790.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1806.40 | 1795.47 | 1792.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 1794.60 | 1798.32 | 1794.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1793.20 | 1798.96 | 1796.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 1792.60 | 1798.96 | 1796.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1794.20 | 1798.01 | 1796.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 1785.00 | 1798.01 | 1796.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1783.50 | 1795.11 | 1795.14 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1797.10 | 1795.02 | 1794.89 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1780.90 | 1792.20 | 1793.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1777.60 | 1784.95 | 1788.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 1788.10 | 1784.05 | 1787.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1790.90 | 1785.42 | 1787.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1790.90 | 1785.42 | 1787.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1749.90 | 1748.01 | 1756.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 1754.00 | 1748.01 | 1756.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1755.70 | 1749.55 | 1756.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1755.70 | 1749.55 | 1756.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 1758.00 | 1751.24 | 1756.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:30:00 | 1758.30 | 1751.24 | 1756.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 1763.80 | 1753.75 | 1757.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 1763.80 | 1753.75 | 1757.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 1771.50 | 1761.32 | 1760.03 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1753.40 | 1759.91 | 1760.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1741.60 | 1756.25 | 1758.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 1725.00 | 1722.09 | 1730.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 1725.00 | 1722.09 | 1730.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1721.00 | 1717.45 | 1721.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 1719.80 | 1717.45 | 1721.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1720.00 | 1717.96 | 1721.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1724.50 | 1717.96 | 1721.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1721.10 | 1718.59 | 1721.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 1713.40 | 1717.97 | 1720.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1725.80 | 1720.83 | 1720.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1725.80 | 1720.83 | 1720.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1731.80 | 1723.03 | 1721.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1734.00 | 1735.42 | 1729.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1734.00 | 1735.42 | 1729.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1730.60 | 1734.45 | 1730.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 1730.60 | 1734.45 | 1730.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1725.70 | 1732.70 | 1729.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1730.40 | 1732.70 | 1729.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1737.30 | 1733.62 | 1730.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1744.60 | 1735.06 | 1731.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1743.80 | 1736.39 | 1732.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 1749.20 | 1760.38 | 1760.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1738.60 | 1753.51 | 1756.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1736.10 | 1731.10 | 1740.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1736.10 | 1731.10 | 1740.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1738.50 | 1732.58 | 1740.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1743.40 | 1732.58 | 1740.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1736.00 | 1733.27 | 1740.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1728.00 | 1733.27 | 1740.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1719.10 | 1730.43 | 1738.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 1714.60 | 1726.26 | 1734.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1715.50 | 1723.23 | 1731.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1712.60 | 1720.82 | 1728.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1628.87 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1629.72 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1626.97 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 1680.10 | 1675.89 | 1689.32 | SL hit (close>ema200) qty=0.50 sl=1675.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 1680.10 | 1675.89 | 1689.32 | SL hit (close>ema200) qty=0.50 sl=1675.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 1680.10 | 1675.89 | 1689.32 | SL hit (close>ema200) qty=0.50 sl=1675.89 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 1644.30 | 1634.90 | 1634.71 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1631.50 | 1634.22 | 1634.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1627.40 | 1631.84 | 1633.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1640.10 | 1633.49 | 1633.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1639.00 | 1634.59 | 1634.30 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 1619.10 | 1631.49 | 1632.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 1615.00 | 1626.05 | 1630.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 12:15:00 | 1596.60 | 1593.79 | 1602.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 1596.60 | 1593.79 | 1602.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1603.60 | 1595.75 | 1603.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 1603.60 | 1595.75 | 1603.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1595.90 | 1595.78 | 1602.37 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 1618.40 | 1606.37 | 1605.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 1630.00 | 1616.45 | 1611.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1693.80 | 1700.61 | 1682.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 12:30:00 | 1703.20 | 1700.15 | 1686.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 14:30:00 | 1704.80 | 1700.90 | 1689.21 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1693.40 | 1699.75 | 1690.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1690.70 | 1699.75 | 1690.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1693.90 | 1697.21 | 1691.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 1693.60 | 1697.21 | 1691.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1707.60 | 1708.18 | 1703.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1707.20 | 1708.18 | 1703.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1707.60 | 1708.06 | 1704.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1704.40 | 1708.06 | 1704.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1706.00 | 1707.65 | 1704.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1711.90 | 1707.65 | 1704.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 1710.70 | 1707.78 | 1704.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 1710.40 | 1708.84 | 1705.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1711.50 | 1708.84 | 1705.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1706.00 | 1709.00 | 1706.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 1706.00 | 1709.00 | 1706.84 | SL hit (close<ema400) qty=1.00 sl=1706.84 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 1706.00 | 1709.00 | 1706.84 | SL hit (close<ema400) qty=1.00 sl=1706.84 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1706.00 | 1709.00 | 1706.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1706.70 | 1708.54 | 1706.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 1715.70 | 1709.92 | 1707.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 1716.20 | 1711.97 | 1709.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1702.80 | 1709.15 | 1709.05 | SL hit (close<static) qty=1.00 sl=1703.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1702.80 | 1709.15 | 1709.05 | SL hit (close<static) qty=1.00 sl=1703.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1702.80 | 1709.15 | 1709.05 | SL hit (close<static) qty=1.00 sl=1703.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1702.80 | 1709.15 | 1709.05 | SL hit (close<static) qty=1.00 sl=1703.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 1698.00 | 1706.92 | 1708.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 1698.00 | 1706.92 | 1708.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1698.00 | 1706.92 | 1708.05 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1715.80 | 1706.75 | 1705.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1717.60 | 1708.92 | 1706.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1720.10 | 1721.79 | 1717.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 1720.10 | 1721.79 | 1717.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1713.00 | 1720.15 | 1718.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1713.00 | 1720.15 | 1718.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1713.00 | 1718.72 | 1717.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1709.60 | 1718.72 | 1717.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1720.40 | 1718.64 | 1717.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 1722.20 | 1720.45 | 1718.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1723.70 | 1722.67 | 1720.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 1742.90 | 1758.08 | 1759.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 1742.90 | 1758.08 | 1759.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1742.90 | 1758.08 | 1759.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1728.20 | 1752.10 | 1756.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1748.40 | 1747.90 | 1752.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:00:00 | 1748.40 | 1747.90 | 1752.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1752.90 | 1748.90 | 1752.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:45:00 | 1752.20 | 1748.90 | 1752.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1753.20 | 1749.76 | 1752.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1739.50 | 1749.76 | 1752.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1779.30 | 1753.70 | 1752.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1779.30 | 1753.70 | 1752.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1783.51 | 1771.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:45:00 | 1799.30 | 1792.65 | 1783.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1801.20 | 1817.22 | 1817.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 1801.20 | 1817.22 | 1817.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1795.00 | 1807.49 | 1812.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1795.90 | 1786.09 | 1796.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1791.00 | 1787.07 | 1795.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 1796.90 | 1787.07 | 1795.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1803.50 | 1789.87 | 1795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 1803.50 | 1789.87 | 1795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1805.40 | 1792.97 | 1796.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1805.40 | 1792.97 | 1796.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1795.50 | 1793.87 | 1796.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 1792.10 | 1794.01 | 1795.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:15:00 | 1791.40 | 1794.01 | 1795.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1782.40 | 1767.10 | 1766.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1782.40 | 1767.10 | 1766.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1782.40 | 1767.10 | 1766.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1791.30 | 1771.94 | 1769.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 1794.70 | 1797.06 | 1786.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 1794.70 | 1797.06 | 1786.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1789.80 | 1795.84 | 1788.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1789.80 | 1795.84 | 1788.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1779.70 | 1792.41 | 1787.79 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1774.40 | 1783.51 | 1784.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1753.40 | 1777.49 | 1781.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1784.00 | 1775.07 | 1779.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1758.30 | 1771.72 | 1777.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:45:00 | 1753.00 | 1763.74 | 1773.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1665.35 | 1728.82 | 1752.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1689.20 | 1685.37 | 1708.70 | SL hit (close>ema200) qty=0.50 sl=1685.37 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 1721.00 | 1710.44 | 1709.96 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1694.10 | 1707.17 | 1708.52 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 1714.10 | 1709.07 | 1708.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 1725.00 | 1712.89 | 1710.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 1713.60 | 1715.79 | 1712.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1716.50 | 1715.93 | 1713.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1713.20 | 1715.93 | 1713.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 09:15:00 | 1656.70 | 1703.94 | 1708.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 10:15:00 | 1640.00 | 1691.15 | 1702.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1659.70 | 1658.94 | 1674.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 1659.70 | 1658.94 | 1674.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1670.00 | 1659.34 | 1668.84 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1689.10 | 1673.68 | 1673.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1697.40 | 1678.43 | 1675.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 1680.40 | 1684.20 | 1679.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1679.30 | 1683.22 | 1679.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 1671.30 | 1683.22 | 1679.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1683.00 | 1683.18 | 1680.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1687.50 | 1684.76 | 1681.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 1689.30 | 1688.21 | 1683.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1676.80 | 1684.77 | 1682.58 | SL hit (close<static) qty=1.00 sl=1677.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1676.80 | 1684.77 | 1682.58 | SL hit (close<static) qty=1.00 sl=1677.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 1675.90 | 1681.11 | 1681.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 11:15:00 | 1670.20 | 1675.55 | 1678.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 1674.30 | 1671.62 | 1674.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1674.80 | 1671.34 | 1673.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 1674.80 | 1671.34 | 1673.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1670.30 | 1671.13 | 1673.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:45:00 | 1675.30 | 1671.13 | 1673.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1664.90 | 1669.89 | 1672.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1656.20 | 1667.50 | 1671.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1659.00 | 1664.82 | 1669.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1695.60 | 1672.45 | 1671.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1695.60 | 1672.45 | 1671.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1695.60 | 1672.45 | 1671.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 1702.10 | 1683.90 | 1677.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1677.70 | 1684.71 | 1678.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1678.50 | 1683.47 | 1678.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1654.00 | 1683.47 | 1678.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1640.00 | 1674.77 | 1675.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1630.60 | 1665.94 | 1671.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1737.50 | 1656.12 | 1660.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1722.30 | 1669.36 | 1665.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 1780.10 | 1758.29 | 1738.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1804.00 | 1812.98 | 1794.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 1804.00 | 1812.98 | 1794.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1829.70 | 1838.86 | 1827.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1829.70 | 1838.86 | 1827.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1845.30 | 1838.75 | 1832.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:30:00 | 1850.40 | 1841.50 | 1834.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1850.10 | 1843.32 | 1835.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:30:00 | 1850.10 | 1844.65 | 1836.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 1851.10 | 1844.65 | 1836.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 10:15:00 | 1693.40 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-05-13 11:00:00 | 1693.90 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-05-14 11:00:00 | 1695.00 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-15 09:30:00 | 1689.70 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-05-20 09:15:00 | 1740.80 | 2025-05-20 12:15:00 | 1722.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1681.10 | 2025-05-29 11:15:00 | 1687.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-05-27 11:45:00 | 1681.30 | 2025-05-29 11:15:00 | 1687.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-05-27 12:15:00 | 1679.60 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-05-27 15:15:00 | 1681.50 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-05-28 14:15:00 | 1667.60 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-28 14:45:00 | 1666.70 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-12 09:15:00 | 1721.80 | 2025-06-13 09:15:00 | 1679.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1665.50 | 2025-06-20 13:15:00 | 1660.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-06-23 12:15:00 | 1660.40 | 2025-06-26 11:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-23 13:15:00 | 1662.20 | 2025-06-26 11:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1681.60 | 2025-07-24 13:15:00 | 1692.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1734.70 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-31 12:00:00 | 1732.90 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1735.00 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1596.40 | 2025-08-11 15:15:00 | 1611.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-25 14:15:00 | 1649.60 | 2025-08-26 09:15:00 | 1619.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-01 09:15:00 | 1590.10 | 2025-09-04 15:15:00 | 1580.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1588.30 | 2025-09-04 15:15:00 | 1580.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-09-10 15:15:00 | 1596.00 | 2025-09-23 10:15:00 | 1630.50 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2025-10-13 12:15:00 | 1665.60 | 2025-10-14 13:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-13 12:45:00 | 1665.10 | 2025-10-14 13:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1695.30 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-24 13:45:00 | 1695.70 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-24 15:00:00 | 1699.30 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-27 10:15:00 | 1693.60 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-28 09:30:00 | 1700.20 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1696.10 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-10-30 12:00:00 | 1695.80 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1697.00 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-10-31 10:30:00 | 1696.00 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-06 10:15:00 | 1668.30 | 2025-11-10 09:15:00 | 1696.40 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-11-11 09:15:00 | 1699.20 | 2025-12-02 09:15:00 | 1797.10 | STOP_HIT | 1.00 | 5.76% |
| BUY | retest2 | 2025-11-11 11:00:00 | 1698.40 | 2025-12-02 09:15:00 | 1797.10 | STOP_HIT | 1.00 | 5.81% |
| SELL | retest2 | 2025-12-03 10:45:00 | 1785.30 | 2025-12-04 11:15:00 | 1806.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1713.40 | 2026-01-02 09:15:00 | 1725.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-01-06 11:15:00 | 1744.60 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1743.80 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2026-01-09 10:45:00 | 1749.20 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1714.60 | 2026-01-19 09:15:00 | 1628.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1715.50 | 2026-01-19 09:15:00 | 1629.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1712.60 | 2026-01-19 09:15:00 | 1626.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1714.60 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1715.50 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1712.60 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2026-02-05 12:30:00 | 1703.20 | 2026-02-11 15:15:00 | 1706.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest1 | 2026-02-05 14:30:00 | 1704.80 | 2026-02-11 15:15:00 | 1706.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1711.90 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-11 11:15:00 | 1710.70 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-11 12:45:00 | 1710.40 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1711.50 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-12 13:15:00 | 1715.70 | 2026-02-13 14:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-12 15:00:00 | 1716.20 | 2026-02-13 14:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-20 13:30:00 | 1722.20 | 2026-02-27 15:15:00 | 1742.90 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2026-02-23 10:30:00 | 1723.70 | 2026-02-27 15:15:00 | 1742.90 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1739.50 | 2026-03-05 09:15:00 | 1779.30 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-09 12:45:00 | 1799.30 | 2026-03-13 12:15:00 | 1801.20 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1792.10 | 2026-03-25 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-03-18 13:15:00 | 1791.40 | 2026-03-25 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1753.00 | 2026-04-02 09:15:00 | 1665.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1753.00 | 2026-04-06 12:15:00 | 1689.20 | STOP_HIT | 0.50 | 3.64% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1687.50 | 2026-04-17 11:15:00 | 1676.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-17 09:45:00 | 1689.30 | 2026-04-17 11:15:00 | 1676.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1656.20 | 2026-04-23 09:15:00 | 1695.60 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-04-22 12:15:00 | 1659.00 | 2026-04-23 09:15:00 | 1695.60 | STOP_HIT | 1.00 | -2.21% |
