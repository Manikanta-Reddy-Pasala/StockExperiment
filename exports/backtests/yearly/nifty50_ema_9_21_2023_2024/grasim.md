# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 2965.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 252 |
| ALERT1 | 171 |
| ALERT2 | 170 |
| ALERT2_SKIP | 88 |
| ALERT3 | 486 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 192 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 205 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 207 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 154
- **Target hits / Stop hits / Partials:** 2 / 197 / 8
- **Avg / median % per leg:** -0.20% / -0.76%
- **Sum % (uncompounded):** -40.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 97 | 23 | 23.7% | 2 | 95 | 0 | -0.32% | -30.8% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.46% | -7.3% |
| BUY @ 3rd Alert (retest2) | 92 | 23 | 25.0% | 2 | 90 | 0 | -0.26% | -23.5% |
| SELL (all) | 110 | 30 | 27.3% | 0 | 102 | 8 | -0.09% | -9.6% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.44% | -0.9% |
| SELL @ 3rd Alert (retest2) | 108 | 29 | 26.9% | 0 | 100 | 8 | -0.08% | -8.7% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.17% | -8.2% |
| retest2 (combined) | 200 | 52 | 26.0% | 2 | 190 | 8 | -0.16% | -32.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-12 10:15:00 | 1752.03 | 1762.05 | 1762.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-12 12:15:00 | 1747.55 | 1757.68 | 1760.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 1728.23 | 1724.38 | 1732.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 10:00:00 | 1728.23 | 1724.38 | 1732.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 1718.17 | 1715.27 | 1723.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 1720.41 | 1715.27 | 1723.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1708.21 | 1713.85 | 1722.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 10:15:00 | 1704.52 | 1713.85 | 1722.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 13:00:00 | 1704.37 | 1705.24 | 1706.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 14:15:00 | 1713.94 | 1707.24 | 1706.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 1713.94 | 1707.24 | 1706.82 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 09:15:00 | 1689.13 | 1704.40 | 1705.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 09:15:00 | 1673.30 | 1690.03 | 1693.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 14:15:00 | 1680.77 | 1679.93 | 1686.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 14:15:00 | 1680.77 | 1679.93 | 1686.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 1680.77 | 1679.93 | 1686.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 14:45:00 | 1680.47 | 1679.93 | 1686.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 1684.75 | 1680.65 | 1685.49 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 14:15:00 | 1705.22 | 1690.71 | 1688.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 09:15:00 | 1720.36 | 1699.44 | 1693.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 11:15:00 | 1713.79 | 1714.28 | 1706.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 12:00:00 | 1713.79 | 1714.28 | 1706.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 1708.91 | 1713.21 | 1707.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 12:30:00 | 1709.30 | 1713.21 | 1707.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 1709.95 | 1712.55 | 1707.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:45:00 | 1709.20 | 1712.55 | 1707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 1693.27 | 1708.87 | 1706.54 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 12:15:00 | 1699.44 | 1705.54 | 1705.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 14:15:00 | 1695.81 | 1702.57 | 1704.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 11:15:00 | 1700.44 | 1700.03 | 1702.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-02 11:45:00 | 1700.09 | 1700.03 | 1702.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 1698.55 | 1699.54 | 1701.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:45:00 | 1700.99 | 1699.54 | 1701.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 1704.37 | 1700.51 | 1701.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 15:00:00 | 1704.37 | 1700.51 | 1701.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 1703.13 | 1701.03 | 1702.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:15:00 | 1712.14 | 1701.03 | 1702.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 1732.96 | 1707.42 | 1704.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 1744.07 | 1726.61 | 1717.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 09:15:00 | 1744.56 | 1747.83 | 1734.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 10:00:00 | 1744.56 | 1747.83 | 1734.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 1728.38 | 1750.02 | 1745.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 1728.93 | 1750.02 | 1745.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 1719.12 | 1743.84 | 1742.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 1719.12 | 1743.84 | 1742.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 13:15:00 | 1717.07 | 1738.49 | 1740.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 14:15:00 | 1704.07 | 1731.60 | 1737.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 11:15:00 | 1730.77 | 1725.44 | 1731.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-09 12:00:00 | 1730.77 | 1725.44 | 1731.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 1725.54 | 1716.78 | 1721.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:00:00 | 1725.54 | 1716.78 | 1721.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 1723.90 | 1718.20 | 1721.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 15:00:00 | 1723.90 | 1718.20 | 1721.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 1725.14 | 1719.59 | 1722.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:15:00 | 1730.12 | 1719.59 | 1722.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 1735.15 | 1724.73 | 1724.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 10:15:00 | 1747.90 | 1734.02 | 1729.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 15:15:00 | 1765.98 | 1767.21 | 1756.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:15:00 | 1779.92 | 1767.21 | 1756.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:45:00 | 1775.39 | 1768.74 | 1759.19 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 13:30:00 | 1772.50 | 1769.60 | 1761.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 1765.48 | 1770.50 | 1765.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-19 11:15:00 | 1758.86 | 1768.17 | 1764.46 | SL hit (close<ema400) qty=1.00 sl=1764.46 alert=retest1 |

### Cycle 9 — SELL (started 2023-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 15:15:00 | 1758.01 | 1762.48 | 1762.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 10:15:00 | 1744.41 | 1758.14 | 1760.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 1763.29 | 1754.95 | 1757.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 1763.29 | 1754.95 | 1757.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 1763.29 | 1754.95 | 1757.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 1763.29 | 1754.95 | 1757.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 1762.99 | 1756.56 | 1758.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 1756.57 | 1756.56 | 1758.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 1752.18 | 1756.71 | 1758.10 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 1763.74 | 1759.27 | 1759.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 13:15:00 | 1765.78 | 1760.57 | 1759.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 1750.34 | 1762.46 | 1761.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 1750.34 | 1762.46 | 1761.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 1750.34 | 1762.46 | 1761.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 1750.34 | 1762.46 | 1761.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 1747.55 | 1759.48 | 1760.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1725.59 | 1746.63 | 1753.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 1708.81 | 1707.69 | 1722.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 1708.81 | 1707.69 | 1722.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 1722.15 | 1710.58 | 1722.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 1722.15 | 1710.58 | 1722.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 1716.18 | 1711.70 | 1721.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 1722.15 | 1711.70 | 1721.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1716.67 | 1712.69 | 1721.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:30:00 | 1714.93 | 1712.31 | 1720.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 14:15:00 | 1726.44 | 1715.28 | 1719.21 | SL hit (close>static) qty=1.00 sl=1725.99 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 1746.11 | 1723.50 | 1722.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 10:15:00 | 1756.12 | 1738.65 | 1734.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 10:15:00 | 1761.15 | 1766.54 | 1753.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 10:30:00 | 1761.80 | 1766.54 | 1753.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 1756.02 | 1762.67 | 1754.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 1756.02 | 1762.67 | 1754.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 1762.94 | 1762.72 | 1755.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 09:15:00 | 1774.30 | 1762.30 | 1756.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:45:00 | 1770.46 | 1763.63 | 1757.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 12:15:00 | 1766.43 | 1763.12 | 1758.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 15:00:00 | 1771.51 | 1766.16 | 1760.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1767.52 | 1767.88 | 1762.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:00:00 | 1767.52 | 1767.88 | 1762.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 1771.46 | 1770.31 | 1765.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 12:45:00 | 1768.77 | 1770.31 | 1765.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 1764.88 | 1769.23 | 1765.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:00:00 | 1764.88 | 1769.23 | 1765.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 1761.75 | 1767.73 | 1764.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:00:00 | 1761.75 | 1767.73 | 1764.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 1760.00 | 1766.18 | 1764.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:15:00 | 1755.12 | 1766.18 | 1764.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 1755.52 | 1764.05 | 1763.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 1756.02 | 1761.95 | 1762.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 1756.02 | 1761.95 | 1762.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 1750.04 | 1759.57 | 1761.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 13:15:00 | 1745.01 | 1741.20 | 1748.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 14:00:00 | 1745.01 | 1741.20 | 1748.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1750.89 | 1742.30 | 1747.38 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 1758.71 | 1751.33 | 1750.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 15:15:00 | 1759.60 | 1754.27 | 1752.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 12:15:00 | 1764.53 | 1767.08 | 1762.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 12:15:00 | 1764.53 | 1767.08 | 1762.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 1764.53 | 1767.08 | 1762.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:45:00 | 1763.34 | 1767.08 | 1762.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 1751.29 | 1763.92 | 1761.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 1751.29 | 1763.92 | 1761.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 1752.93 | 1761.72 | 1760.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 1755.52 | 1759.98 | 1759.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 09:15:00 | 1748.80 | 1757.75 | 1758.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 09:15:00 | 1748.80 | 1757.75 | 1758.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 1744.71 | 1753.53 | 1756.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 1751.98 | 1750.55 | 1753.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 1751.98 | 1750.55 | 1753.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1751.98 | 1750.55 | 1753.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 1751.09 | 1750.55 | 1753.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 1746.61 | 1749.65 | 1752.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:30:00 | 1750.04 | 1749.65 | 1752.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 1749.39 | 1749.60 | 1752.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:00:00 | 1749.39 | 1749.60 | 1752.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 1772.05 | 1754.09 | 1754.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:00:00 | 1772.05 | 1754.09 | 1754.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 14:15:00 | 1786.70 | 1760.61 | 1757.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 11:15:00 | 1801.14 | 1779.73 | 1773.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 1797.01 | 1802.97 | 1793.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 15:00:00 | 1797.01 | 1802.97 | 1793.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1792.97 | 1800.97 | 1793.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 1804.97 | 1800.97 | 1793.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1823.10 | 1805.40 | 1796.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 10:15:00 | 1831.47 | 1805.40 | 1796.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 1831.42 | 1808.90 | 1802.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:30:00 | 1840.58 | 1820.08 | 1812.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 11:15:00 | 1831.12 | 1821.36 | 1813.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1822.75 | 1827.83 | 1821.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 1822.75 | 1827.83 | 1821.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 1818.52 | 1825.96 | 1821.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:00:00 | 1818.52 | 1825.96 | 1821.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 1813.39 | 1823.45 | 1820.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 12:00:00 | 1813.39 | 1823.45 | 1820.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 1818.32 | 1821.90 | 1820.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:45:00 | 1818.77 | 1821.90 | 1820.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 1815.73 | 1820.66 | 1819.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 1808.46 | 1820.66 | 1819.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-27 15:15:00 | 1811.80 | 1818.89 | 1819.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 15:15:00 | 1811.80 | 1818.89 | 1819.00 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1841.68 | 1820.39 | 1818.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 1844.42 | 1831.25 | 1825.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 1827.68 | 1832.26 | 1827.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 1827.68 | 1832.26 | 1827.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1827.68 | 1832.26 | 1827.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 1827.68 | 1832.26 | 1827.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 1822.11 | 1830.23 | 1826.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:00:00 | 1822.11 | 1830.23 | 1826.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 1820.41 | 1828.27 | 1826.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:00:00 | 1820.41 | 1828.27 | 1826.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 1830.67 | 1828.60 | 1826.77 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 1811.90 | 1823.43 | 1824.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 1807.31 | 1820.21 | 1823.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 1815.53 | 1814.95 | 1819.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 15:00:00 | 1815.53 | 1814.95 | 1819.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 1816.88 | 1813.67 | 1817.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:00:00 | 1816.88 | 1813.67 | 1817.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 1797.60 | 1810.46 | 1815.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 12:30:00 | 1790.53 | 1809.05 | 1814.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 13:15:00 | 1817.77 | 1810.80 | 1815.08 | SL hit (close>static) qty=1.00 sl=1816.93 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 1827.73 | 1818.86 | 1817.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 1841.18 | 1825.02 | 1821.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 1839.69 | 1840.92 | 1833.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 1839.69 | 1840.92 | 1833.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 1839.19 | 1840.58 | 1834.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 1834.51 | 1840.58 | 1834.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 1833.71 | 1839.20 | 1834.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:00:00 | 1833.71 | 1839.20 | 1834.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 1826.99 | 1836.76 | 1833.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:00:00 | 1826.99 | 1836.76 | 1833.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 1824.84 | 1834.36 | 1833.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:30:00 | 1824.99 | 1834.36 | 1833.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 1827.73 | 1833.03 | 1832.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:15:00 | 1822.75 | 1833.03 | 1832.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 1828.63 | 1832.15 | 1832.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 12:15:00 | 1821.56 | 1830.03 | 1831.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 13:15:00 | 1832.36 | 1830.50 | 1831.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 13:15:00 | 1832.36 | 1830.50 | 1831.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 1832.36 | 1830.50 | 1831.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:00:00 | 1832.36 | 1830.50 | 1831.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 1837.99 | 1832.00 | 1832.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 1837.99 | 1832.00 | 1832.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 1839.89 | 1833.58 | 1832.75 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 12:15:00 | 1825.94 | 1832.49 | 1832.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 1817.47 | 1825.27 | 1828.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 14:15:00 | 1820.61 | 1819.91 | 1824.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 14:15:00 | 1820.61 | 1819.91 | 1824.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 1820.61 | 1819.91 | 1824.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 09:15:00 | 1793.52 | 1819.56 | 1823.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 14:15:00 | 1800.84 | 1795.57 | 1795.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 1800.84 | 1795.57 | 1795.45 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 11:15:00 | 1792.87 | 1796.73 | 1796.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 12:15:00 | 1788.24 | 1795.04 | 1795.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 13:15:00 | 1797.30 | 1795.49 | 1796.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 13:15:00 | 1797.30 | 1795.49 | 1796.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 1797.30 | 1795.49 | 1796.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 1797.30 | 1795.49 | 1796.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 1789.73 | 1794.34 | 1795.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:45:00 | 1787.09 | 1793.74 | 1794.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:15:00 | 1783.96 | 1793.74 | 1794.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 1786.20 | 1776.87 | 1776.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 1786.20 | 1776.87 | 1776.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 1790.33 | 1779.56 | 1777.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 1802.28 | 1802.67 | 1793.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 15:00:00 | 1802.28 | 1802.67 | 1793.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1804.08 | 1802.30 | 1794.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 1795.61 | 1802.30 | 1794.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 1787.44 | 1798.51 | 1794.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:00:00 | 1787.44 | 1798.51 | 1794.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 1784.21 | 1795.65 | 1793.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 13:00:00 | 1784.21 | 1795.65 | 1793.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 1782.91 | 1790.93 | 1791.65 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 1799.99 | 1793.35 | 1792.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 15:15:00 | 1817.72 | 1799.11 | 1795.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 1832.27 | 1840.43 | 1830.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 09:45:00 | 1831.77 | 1840.43 | 1830.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 1830.82 | 1838.51 | 1830.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:00:00 | 1830.82 | 1838.51 | 1830.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 1827.09 | 1836.23 | 1829.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 1824.30 | 1836.23 | 1829.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 1816.58 | 1832.30 | 1828.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:00:00 | 1816.58 | 1832.30 | 1828.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 1823.10 | 1827.85 | 1827.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 1823.10 | 1827.85 | 1827.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 1826.94 | 1827.67 | 1827.33 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 12:15:00 | 1823.70 | 1826.54 | 1826.86 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 1837.69 | 1829.13 | 1828.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 1843.12 | 1836.03 | 1832.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 15:15:00 | 1847.65 | 1849.07 | 1843.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:15:00 | 1858.11 | 1849.07 | 1843.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1847.65 | 1848.79 | 1843.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1838.89 | 1848.79 | 1843.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1887.94 | 1856.62 | 1847.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 1847.55 | 1856.62 | 1847.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1883.26 | 1870.86 | 1860.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:30:00 | 1927.94 | 1887.50 | 1869.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 12:15:00 | 1928.93 | 1887.50 | 1869.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 13:00:00 | 1923.35 | 1894.67 | 1874.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 14:00:00 | 1923.35 | 1900.41 | 1879.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1938.49 | 1948.98 | 1941.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-20 15:15:00 | 1933.56 | 1938.63 | 1939.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 1933.56 | 1938.63 | 1939.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 1905.25 | 1931.95 | 1936.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 15:15:00 | 1905.62 | 1905.37 | 1913.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 09:15:00 | 1906.92 | 1905.37 | 1913.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1911.15 | 1906.53 | 1913.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:00:00 | 1911.15 | 1906.53 | 1913.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 1914.59 | 1908.14 | 1913.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 1917.08 | 1908.14 | 1913.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 1916.58 | 1909.83 | 1913.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:30:00 | 1914.34 | 1909.83 | 1913.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 1917.33 | 1911.33 | 1914.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 1917.33 | 1911.33 | 1914.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 1932.07 | 1916.45 | 1916.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 15:15:00 | 1937.25 | 1920.61 | 1917.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 1938.04 | 1940.20 | 1931.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 10:00:00 | 1938.04 | 1940.20 | 1931.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 1934.01 | 1938.96 | 1932.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:45:00 | 1931.92 | 1938.96 | 1932.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1932.32 | 1937.63 | 1932.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:30:00 | 1930.38 | 1937.63 | 1932.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 1927.29 | 1935.56 | 1931.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:45:00 | 1925.54 | 1935.56 | 1931.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 13:15:00 | 1919.27 | 1932.30 | 1930.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 13:45:00 | 1922.31 | 1932.30 | 1930.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 15:15:00 | 1924.75 | 1929.33 | 1929.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 1915.83 | 1926.63 | 1928.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1920.37 | 1912.35 | 1918.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 1920.37 | 1912.35 | 1918.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 1920.37 | 1912.35 | 1918.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 1920.37 | 1912.35 | 1918.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 1925.44 | 1914.97 | 1918.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 11:00:00 | 1925.44 | 1914.97 | 1918.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 11:15:00 | 1933.01 | 1918.58 | 1920.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 11:45:00 | 1932.22 | 1918.58 | 1920.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 1933.31 | 1921.52 | 1921.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 13:15:00 | 1942.28 | 1925.68 | 1923.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 1909.11 | 1925.33 | 1924.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 1909.11 | 1925.33 | 1924.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1909.11 | 1925.33 | 1924.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 1909.11 | 1925.33 | 1924.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 10:15:00 | 1908.41 | 1921.95 | 1922.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 1889.79 | 1909.46 | 1915.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1893.22 | 1890.86 | 1900.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1893.22 | 1890.86 | 1900.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1893.22 | 1890.86 | 1900.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:00:00 | 1885.70 | 1889.83 | 1899.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 1874.00 | 1892.19 | 1893.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 10:30:00 | 1886.00 | 1890.55 | 1892.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:00:00 | 1886.80 | 1890.55 | 1892.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1884.66 | 1882.91 | 1886.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-10 13:15:00 | 1903.68 | 1891.23 | 1889.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 1903.68 | 1891.23 | 1889.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 1911.05 | 1895.19 | 1891.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 1977.14 | 1982.32 | 1962.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 10:45:00 | 1977.14 | 1982.32 | 1962.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1959.71 | 1972.77 | 1965.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 10:00:00 | 1959.71 | 1972.77 | 1965.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 1952.99 | 1968.81 | 1964.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 11:00:00 | 1952.99 | 1968.81 | 1964.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 13:15:00 | 1965.19 | 1967.13 | 1964.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 1981.37 | 1966.07 | 1964.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 11:15:00 | 1959.16 | 1964.67 | 1964.30 | SL hit (close<static) qty=1.00 sl=1964.24 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 12:15:00 | 1952.79 | 1962.30 | 1963.25 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 1977.09 | 1966.01 | 1964.63 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 1955.23 | 1962.57 | 1963.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 1939.19 | 1953.16 | 1958.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 14:15:00 | 1967.88 | 1948.02 | 1953.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 1967.88 | 1948.02 | 1953.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1967.88 | 1948.02 | 1953.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 1968.03 | 1948.02 | 1953.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 1963.00 | 1951.01 | 1954.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:30:00 | 1956.22 | 1952.28 | 1954.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1858.41 | 1880.72 | 1900.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 11:15:00 | 1856.57 | 1854.33 | 1871.08 | SL hit (close>ema200) qty=0.50 sl=1854.33 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 1875.05 | 1865.73 | 1864.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 1878.23 | 1868.23 | 1865.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 14:15:00 | 1860.90 | 1874.25 | 1871.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 14:15:00 | 1860.90 | 1874.25 | 1871.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 1860.90 | 1874.25 | 1871.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:00:00 | 1860.90 | 1874.25 | 1871.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 1864.59 | 1872.32 | 1870.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 1882.71 | 1872.32 | 1870.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 15:15:00 | 1914.39 | 1923.60 | 1924.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 15:15:00 | 1914.39 | 1923.60 | 1924.77 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1937.20 | 1926.32 | 1925.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 09:15:00 | 1958.26 | 1937.22 | 1933.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 10:15:00 | 1948.95 | 1956.13 | 1948.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 10:15:00 | 1948.95 | 1956.13 | 1948.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1948.95 | 1956.13 | 1948.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 1948.95 | 1956.13 | 1948.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 1953.23 | 1955.55 | 1948.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 12:30:00 | 1956.22 | 1957.05 | 1950.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 1960.16 | 1963.29 | 1961.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 13:15:00 | 1952.89 | 1960.88 | 1961.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 13:15:00 | 1952.89 | 1960.88 | 1961.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 14:15:00 | 1951.14 | 1958.93 | 1960.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 1954.63 | 1948.46 | 1952.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 1954.63 | 1948.46 | 1952.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1954.63 | 1948.46 | 1952.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:45:00 | 1957.22 | 1948.46 | 1952.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 1952.19 | 1949.21 | 1952.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 1955.67 | 1949.21 | 1952.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 1953.13 | 1949.99 | 1952.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 13:00:00 | 1951.99 | 1950.39 | 1952.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 14:15:00 | 1968.77 | 1954.15 | 1953.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 1968.77 | 1954.15 | 1953.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 15:15:00 | 1979.13 | 1959.14 | 1956.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 1987.30 | 1989.81 | 1979.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 13:00:00 | 1987.30 | 1989.81 | 1979.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 2039.99 | 2051.29 | 2040.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 2039.99 | 2051.29 | 2040.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 2051.05 | 2051.24 | 2041.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 10:45:00 | 2054.18 | 2051.96 | 2045.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 12:30:00 | 2054.53 | 2052.76 | 2046.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 14:30:00 | 2052.79 | 2059.48 | 2055.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 10:15:00 | 2041.88 | 2067.53 | 2068.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 2041.88 | 2067.53 | 2068.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 2033.52 | 2060.73 | 2065.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 2071.02 | 2062.11 | 2064.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 13:15:00 | 2071.02 | 2062.11 | 2064.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 2071.02 | 2062.11 | 2064.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 14:00:00 | 2071.02 | 2062.11 | 2064.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 2077.34 | 2065.16 | 2066.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 2077.34 | 2065.16 | 2066.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 2067.93 | 2065.71 | 2066.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 2083.42 | 2065.71 | 2066.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 2088.50 | 2070.27 | 2068.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 2093.78 | 2081.23 | 2074.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 2103.79 | 2104.60 | 2093.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 09:45:00 | 2105.63 | 2104.60 | 2093.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 2098.31 | 2102.04 | 2095.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:30:00 | 2096.81 | 2102.04 | 2095.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 2082.82 | 2097.32 | 2095.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 2082.82 | 2097.32 | 2095.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 2092.28 | 2096.31 | 2094.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:15:00 | 2096.02 | 2096.31 | 2094.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 12:00:00 | 2106.97 | 2103.85 | 2101.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 2084.01 | 2098.58 | 2099.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 2084.01 | 2098.58 | 2099.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 2052.29 | 2089.32 | 2094.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 2077.49 | 2075.69 | 2084.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:30:00 | 2073.85 | 2075.69 | 2084.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 2068.13 | 2074.85 | 2081.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 10:45:00 | 2064.49 | 2073.24 | 2080.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 11:15:00 | 2064.79 | 2073.24 | 2080.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:00:00 | 2064.14 | 2071.42 | 2078.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:45:00 | 2058.42 | 2068.20 | 2076.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 2060.66 | 2056.53 | 2066.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 12:15:00 | 2057.97 | 2058.56 | 2066.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:00:00 | 2055.58 | 2057.96 | 2065.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:30:00 | 2054.98 | 2059.53 | 2065.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 15:15:00 | 2071.71 | 2062.01 | 2065.44 | SL hit (close>static) qty=1.00 sl=2071.66 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 2086.70 | 2068.35 | 2067.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 12:15:00 | 2095.92 | 2076.48 | 2071.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 12:15:00 | 2110.76 | 2113.71 | 2103.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 12:45:00 | 2107.57 | 2113.71 | 2103.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 2118.68 | 2133.39 | 2122.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 2118.68 | 2133.39 | 2122.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 2110.51 | 2128.82 | 2121.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 2106.87 | 2123.43 | 2119.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 2082.72 | 2115.29 | 2116.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 2075.05 | 2096.17 | 2105.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 2086.55 | 2079.27 | 2090.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-04 10:00:00 | 2086.55 | 2079.27 | 2090.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 2066.28 | 2067.83 | 2078.20 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 13:15:00 | 2077.24 | 2069.50 | 2068.93 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 2053.69 | 2066.34 | 2067.54 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 12:15:00 | 2082.30 | 2065.74 | 2064.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 13:15:00 | 2087.80 | 2070.16 | 2066.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 10:15:00 | 2095.55 | 2097.81 | 2087.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 11:00:00 | 2095.55 | 2097.81 | 2087.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 2107.70 | 2102.07 | 2094.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 2108.55 | 2102.07 | 2094.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 14:00:00 | 2108.85 | 2103.56 | 2097.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 15:00:00 | 2114.80 | 2105.80 | 2099.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 10:15:00 | 2076.65 | 2098.31 | 2097.42 | SL hit (close<static) qty=1.00 sl=2092.55 alert=retest2 |

### Cycle 53 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 2065.30 | 2091.71 | 2094.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 2050.30 | 2073.31 | 2083.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 15:15:00 | 2065.95 | 2063.28 | 2073.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:15:00 | 2065.70 | 2063.28 | 2073.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 2063.35 | 2063.52 | 2071.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:45:00 | 2065.00 | 2063.52 | 2071.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 2082.95 | 2066.17 | 2071.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 13:00:00 | 2082.95 | 2066.17 | 2071.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 13:15:00 | 2082.65 | 2069.47 | 2072.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:45:00 | 2076.55 | 2070.96 | 2072.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 15:15:00 | 2091.00 | 2074.97 | 2074.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 2091.00 | 2074.97 | 2074.44 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 2055.45 | 2073.15 | 2074.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 2047.50 | 2065.37 | 2070.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 2068.00 | 2060.77 | 2066.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 2068.00 | 2060.77 | 2066.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 2068.00 | 2060.77 | 2066.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 2061.65 | 2060.77 | 2066.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 2056.90 | 2059.99 | 2065.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:30:00 | 2053.60 | 2060.39 | 2065.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 15:15:00 | 2080.00 | 2067.20 | 2067.30 | SL hit (close>static) qty=1.00 sl=2074.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 2071.85 | 2068.13 | 2067.72 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 2063.55 | 2067.22 | 2067.34 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 2069.70 | 2067.71 | 2067.55 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 13:15:00 | 2059.15 | 2066.26 | 2066.94 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 2074.55 | 2067.92 | 2067.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 15:15:00 | 2087.00 | 2071.74 | 2069.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 2142.70 | 2149.14 | 2130.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 09:45:00 | 2147.50 | 2149.14 | 2130.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 2140.00 | 2147.29 | 2133.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 2140.00 | 2147.29 | 2133.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 2137.10 | 2144.05 | 2134.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 2134.50 | 2144.05 | 2134.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 2127.40 | 2140.72 | 2133.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:00:00 | 2127.40 | 2140.72 | 2133.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 2118.35 | 2136.25 | 2132.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 2144.00 | 2136.25 | 2132.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 10:45:00 | 2130.25 | 2137.12 | 2136.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 11:15:00 | 2115.00 | 2132.70 | 2134.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 2115.00 | 2132.70 | 2134.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 2112.10 | 2128.58 | 2132.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 2087.40 | 2083.52 | 2099.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-07 09:30:00 | 2087.50 | 2083.52 | 2099.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 2100.00 | 2086.82 | 2099.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:00:00 | 2100.00 | 2086.82 | 2099.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 2097.55 | 2088.96 | 2099.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:15:00 | 2098.70 | 2088.96 | 2099.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 2100.10 | 2091.19 | 2099.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:30:00 | 2094.35 | 2097.68 | 2100.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 13:30:00 | 2094.20 | 2087.52 | 2094.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 11:15:00 | 2122.50 | 2083.92 | 2088.66 | SL hit (close>static) qty=1.00 sl=2109.85 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 12:15:00 | 2152.90 | 2097.72 | 2094.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 15:15:00 | 2179.05 | 2132.70 | 2112.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 2147.95 | 2152.54 | 2134.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 15:00:00 | 2147.95 | 2152.54 | 2134.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 2114.70 | 2142.97 | 2132.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:45:00 | 2118.45 | 2142.97 | 2132.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 2088.05 | 2131.99 | 2128.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 2088.05 | 2131.99 | 2128.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 2087.65 | 2123.12 | 2125.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 12:15:00 | 2063.25 | 2111.15 | 2119.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 14:15:00 | 2076.20 | 2067.21 | 2084.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 14:15:00 | 2076.20 | 2067.21 | 2084.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 2076.20 | 2067.21 | 2084.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 2076.20 | 2067.21 | 2084.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 2085.55 | 2072.41 | 2083.96 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 2107.90 | 2088.94 | 2087.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 11:15:00 | 2117.65 | 2097.56 | 2092.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 2195.05 | 2197.62 | 2177.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-21 14:00:00 | 2195.05 | 2197.62 | 2177.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 2128.00 | 2182.13 | 2176.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 2128.00 | 2182.13 | 2176.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 2150.75 | 2175.85 | 2174.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 12:15:00 | 2166.05 | 2175.85 | 2174.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 12:15:00 | 2143.75 | 2169.43 | 2171.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 12:15:00 | 2143.75 | 2169.43 | 2171.80 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 2199.45 | 2176.32 | 2174.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 2209.90 | 2198.72 | 2193.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 09:15:00 | 2199.95 | 2200.34 | 2195.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 2199.95 | 2200.34 | 2195.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 2199.95 | 2200.34 | 2195.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 10:15:00 | 2214.00 | 2200.34 | 2195.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 2189.95 | 2198.26 | 2194.67 | SL hit (close<static) qty=1.00 sl=2192.85 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 15:15:00 | 2175.00 | 2190.24 | 2191.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 2160.90 | 2184.37 | 2189.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 2192.80 | 2185.01 | 2188.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 2192.80 | 2185.01 | 2188.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 2192.80 | 2185.01 | 2188.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:30:00 | 2196.65 | 2185.01 | 2188.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 2188.40 | 2185.68 | 2188.51 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 2205.00 | 2190.66 | 2190.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 2214.15 | 2195.36 | 2192.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 2232.85 | 2239.68 | 2225.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 10:00:00 | 2232.85 | 2239.68 | 2225.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 2231.30 | 2237.20 | 2227.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:45:00 | 2230.25 | 2237.20 | 2227.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 2234.20 | 2235.64 | 2228.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 2232.05 | 2235.64 | 2228.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 2240.70 | 2236.84 | 2230.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:15:00 | 2246.05 | 2236.06 | 2230.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:45:00 | 2248.45 | 2239.57 | 2233.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 14:30:00 | 2243.30 | 2238.18 | 2233.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 10:15:00 | 2209.90 | 2228.67 | 2230.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 2209.90 | 2228.67 | 2230.12 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 2242.25 | 2229.75 | 2228.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 2244.90 | 2232.78 | 2230.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 2233.75 | 2237.27 | 2234.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 2233.75 | 2237.27 | 2234.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 2233.75 | 2237.27 | 2234.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:00:00 | 2233.75 | 2237.27 | 2234.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 2234.00 | 2236.61 | 2234.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:15:00 | 2216.20 | 2236.61 | 2234.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 2211.05 | 2231.50 | 2232.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 2206.40 | 2222.80 | 2227.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 2172.60 | 2165.80 | 2185.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 2172.60 | 2165.80 | 2185.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 2187.10 | 2170.06 | 2185.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 2187.10 | 2170.06 | 2185.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 2190.35 | 2174.12 | 2186.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:45:00 | 2188.00 | 2174.12 | 2186.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 2191.00 | 2177.49 | 2186.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 2185.75 | 2177.49 | 2186.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 2185.20 | 2179.03 | 2186.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:45:00 | 2170.75 | 2183.26 | 2186.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 2169.85 | 2183.26 | 2186.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:00:00 | 2177.30 | 2182.07 | 2186.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:45:00 | 2176.85 | 2181.52 | 2185.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 2191.90 | 2183.60 | 2186.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:45:00 | 2190.00 | 2183.60 | 2186.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 2190.00 | 2184.88 | 2186.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:30:00 | 2200.45 | 2184.88 | 2186.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-15 14:15:00 | 2197.45 | 2187.39 | 2187.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 14:15:00 | 2197.45 | 2187.39 | 2187.37 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 15:15:00 | 2169.60 | 2183.83 | 2185.76 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 2202.00 | 2189.74 | 2188.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 12:15:00 | 2206.10 | 2193.01 | 2189.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 2180.40 | 2194.30 | 2191.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 2180.40 | 2194.30 | 2191.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 2180.40 | 2194.30 | 2191.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 2180.40 | 2194.30 | 2191.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 2173.25 | 2190.09 | 2190.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 2149.20 | 2175.49 | 2182.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 11:15:00 | 2179.25 | 2175.51 | 2181.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 11:15:00 | 2179.25 | 2175.51 | 2181.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 2179.25 | 2175.51 | 2181.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:30:00 | 2180.00 | 2175.51 | 2181.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 2186.30 | 2177.66 | 2181.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 2186.30 | 2177.66 | 2181.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 2194.70 | 2181.07 | 2182.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 14:15:00 | 2181.55 | 2181.07 | 2182.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 09:45:00 | 2181.55 | 2179.74 | 2181.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 2198.95 | 2183.58 | 2183.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 2198.95 | 2183.58 | 2183.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 2200.00 | 2189.21 | 2186.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 13:15:00 | 2225.00 | 2233.15 | 2224.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 13:15:00 | 2225.00 | 2233.15 | 2224.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 2225.00 | 2233.15 | 2224.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:00:00 | 2225.00 | 2233.15 | 2224.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 2206.00 | 2227.72 | 2222.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 2206.00 | 2227.72 | 2222.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 2215.00 | 2225.18 | 2221.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 2223.60 | 2225.18 | 2221.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 2268.05 | 2289.25 | 2290.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 2268.05 | 2289.25 | 2290.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 12:15:00 | 2261.30 | 2283.66 | 2287.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 15:15:00 | 2285.00 | 2282.23 | 2285.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:15:00 | 2270.55 | 2282.23 | 2285.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 2264.40 | 2253.26 | 2263.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-08 10:15:00 | 2264.40 | 2253.26 | 2263.69 | SL hit (close>ema400) qty=1.00 sl=2263.69 alert=retest1 |

### Cycle 78 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 2284.00 | 2268.24 | 2267.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 2297.00 | 2273.99 | 2270.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 13:15:00 | 2300.85 | 2301.08 | 2291.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-10 13:45:00 | 2305.25 | 2301.08 | 2291.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 2302.20 | 2301.93 | 2293.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 2282.45 | 2301.93 | 2293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 2287.05 | 2298.95 | 2293.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 2271.60 | 2298.95 | 2293.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 2300.00 | 2299.16 | 2293.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:45:00 | 2292.85 | 2299.16 | 2293.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 2282.65 | 2296.19 | 2293.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 2282.65 | 2296.19 | 2293.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 2272.85 | 2291.52 | 2291.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 2257.50 | 2284.72 | 2288.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 2244.70 | 2243.16 | 2254.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 14:15:00 | 2239.25 | 2243.16 | 2254.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2264.80 | 2247.47 | 2253.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 2264.80 | 2247.47 | 2253.81 | SL hit (close>ema400) qty=1.00 sl=2253.81 alert=retest1 |

### Cycle 80 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 2281.30 | 2259.80 | 2258.66 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 2223.00 | 2254.75 | 2257.01 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 2271.80 | 2258.76 | 2257.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 2278.00 | 2262.61 | 2259.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 2354.00 | 2355.11 | 2328.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 2354.00 | 2355.11 | 2328.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 2370.40 | 2352.79 | 2335.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:30:00 | 2365.00 | 2352.79 | 2335.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 2344.00 | 2356.42 | 2346.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 2344.00 | 2356.42 | 2346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 2339.35 | 2353.00 | 2345.74 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 2322.95 | 2341.15 | 2342.60 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 11:15:00 | 2365.00 | 2346.87 | 2345.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 12:15:00 | 2376.30 | 2352.76 | 2347.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 09:15:00 | 2447.85 | 2462.06 | 2441.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 10:00:00 | 2447.85 | 2462.06 | 2441.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 2459.30 | 2461.51 | 2442.70 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 2411.35 | 2439.80 | 2441.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 15:15:00 | 2404.00 | 2427.14 | 2435.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 13:15:00 | 2372.30 | 2353.02 | 2370.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 13:15:00 | 2372.30 | 2353.02 | 2370.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 2372.30 | 2353.02 | 2370.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 2375.60 | 2353.02 | 2370.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 2375.50 | 2357.51 | 2371.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 2373.75 | 2357.51 | 2371.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 2362.00 | 2358.41 | 2370.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 2368.60 | 2358.41 | 2370.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 2359.10 | 2358.55 | 2369.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 2350.00 | 2357.14 | 2367.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 15:15:00 | 2392.00 | 2376.11 | 2374.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 2392.00 | 2376.11 | 2374.12 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 15:15:00 | 2370.00 | 2375.20 | 2375.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 2341.15 | 2367.73 | 2371.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2370.65 | 2355.13 | 2363.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2370.65 | 2355.13 | 2363.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2379.00 | 2359.90 | 2365.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 2409.25 | 2359.90 | 2365.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 2415.60 | 2377.16 | 2372.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 14:15:00 | 2421.15 | 2399.06 | 2385.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 2430.55 | 2435.68 | 2421.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:45:00 | 2438.80 | 2435.68 | 2421.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2424.00 | 2433.35 | 2421.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 2424.00 | 2433.35 | 2421.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2431.95 | 2433.07 | 2422.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:30:00 | 2437.40 | 2432.87 | 2424.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 2404.00 | 2427.38 | 2423.75 | SL hit (close<static) qty=1.00 sl=2420.95 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 2413.05 | 2420.96 | 2421.24 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 2434.55 | 2423.68 | 2422.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 2446.20 | 2428.18 | 2424.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 2440.05 | 2449.92 | 2440.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 2440.05 | 2449.92 | 2440.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 2443.65 | 2448.66 | 2441.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 2437.85 | 2448.66 | 2441.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2420.20 | 2442.97 | 2439.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 2420.20 | 2442.97 | 2439.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 2407.00 | 2435.78 | 2436.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 2392.10 | 2422.00 | 2429.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 2427.30 | 2410.42 | 2418.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 2421.45 | 2410.42 | 2418.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 2428.45 | 2414.02 | 2419.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 2428.45 | 2414.02 | 2419.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 2441.00 | 2419.42 | 2421.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:30:00 | 2441.45 | 2419.42 | 2421.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 2440.50 | 2423.63 | 2423.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 2455.80 | 2435.08 | 2428.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 2423.80 | 2436.59 | 2430.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 2423.80 | 2436.59 | 2430.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 2399.40 | 2429.16 | 2428.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 2399.40 | 2429.16 | 2428.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 15:15:00 | 2399.00 | 2423.12 | 2425.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 2366.85 | 2411.87 | 2420.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2404.85 | 2351.08 | 2365.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 2404.85 | 2351.08 | 2365.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2390.45 | 2358.95 | 2367.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:45:00 | 2384.95 | 2363.81 | 2369.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 2265.70 | 2329.30 | 2351.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 12:15:00 | 2262.20 | 2258.96 | 2295.57 | SL hit (close>ema200) qty=0.50 sl=2258.96 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 2322.40 | 2307.72 | 2306.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 2335.30 | 2313.23 | 2309.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 2448.10 | 2452.04 | 2424.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 09:15:00 | 2448.50 | 2452.04 | 2424.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2436.10 | 2451.07 | 2440.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 2436.10 | 2451.07 | 2440.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 2433.15 | 2447.48 | 2439.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 2433.15 | 2447.48 | 2439.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 2437.95 | 2445.58 | 2439.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:30:00 | 2433.60 | 2445.58 | 2439.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 2446.00 | 2445.66 | 2440.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 12:45:00 | 2441.85 | 2445.66 | 2440.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 2445.85 | 2445.70 | 2440.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:45:00 | 2447.80 | 2445.70 | 2440.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 2460.85 | 2448.73 | 2442.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:30:00 | 2450.70 | 2448.73 | 2442.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 2518.00 | 2463.75 | 2450.43 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 2422.00 | 2452.86 | 2456.75 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 2467.00 | 2454.70 | 2453.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 13:15:00 | 2497.95 | 2463.35 | 2457.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 2476.30 | 2479.36 | 2469.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 12:00:00 | 2476.30 | 2479.36 | 2469.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 2464.00 | 2476.29 | 2468.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 2464.00 | 2476.29 | 2468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 2481.30 | 2477.29 | 2470.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 2469.80 | 2477.29 | 2470.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 2462.40 | 2474.31 | 2469.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 2462.40 | 2474.31 | 2469.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 2468.00 | 2473.05 | 2469.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 2461.45 | 2473.05 | 2469.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 2482.20 | 2474.88 | 2470.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:30:00 | 2483.45 | 2477.42 | 2471.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:00:00 | 2487.60 | 2477.42 | 2471.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-01 14:15:00 | 2731.80 | 2695.86 | 2655.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 2720.00 | 2732.64 | 2734.02 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 2738.30 | 2734.95 | 2734.91 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 2732.20 | 2734.40 | 2734.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 2723.00 | 2731.95 | 2733.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 2736.25 | 2731.92 | 2733.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 2736.25 | 2731.92 | 2733.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 2752.10 | 2735.95 | 2734.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 15:15:00 | 2784.80 | 2745.72 | 2739.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 2793.75 | 2801.82 | 2780.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 13:00:00 | 2793.75 | 2801.82 | 2780.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2809.00 | 2827.43 | 2811.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2796.00 | 2827.43 | 2811.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2804.00 | 2822.74 | 2810.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 2805.00 | 2822.74 | 2810.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 2795.05 | 2817.20 | 2809.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 2793.45 | 2817.20 | 2809.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 2798.50 | 2812.76 | 2808.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 2798.50 | 2812.76 | 2808.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2803.80 | 2810.97 | 2808.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 2822.25 | 2809.26 | 2807.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 11:15:00 | 2792.60 | 2815.61 | 2815.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 2792.60 | 2815.61 | 2815.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 2762.00 | 2794.82 | 2804.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2796.00 | 2769.65 | 2783.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 2796.00 | 2769.65 | 2783.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2797.00 | 2775.12 | 2784.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 2797.00 | 2775.12 | 2784.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2811.95 | 2784.05 | 2786.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 2816.40 | 2784.05 | 2786.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 2803.65 | 2787.97 | 2788.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 2817.40 | 2787.97 | 2788.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 2813.90 | 2793.16 | 2790.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 2841.50 | 2806.74 | 2797.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2782.00 | 2811.40 | 2805.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 2789.10 | 2811.40 | 2805.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2794.10 | 2807.94 | 2804.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 2775.10 | 2807.94 | 2804.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 2782.70 | 2799.99 | 2801.44 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 2821.00 | 2805.65 | 2803.85 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 2796.40 | 2802.23 | 2802.70 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 2829.90 | 2806.68 | 2804.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 2851.75 | 2815.69 | 2808.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 2833.00 | 2837.80 | 2825.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 2838.90 | 2837.80 | 2825.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 2837.85 | 2837.81 | 2826.51 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 2807.30 | 2822.89 | 2824.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 2796.55 | 2817.62 | 2821.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2660.00 | 2645.48 | 2686.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:45:00 | 2661.35 | 2645.48 | 2686.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 2610.95 | 2583.41 | 2610.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:00:00 | 2610.95 | 2583.41 | 2610.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 2608.35 | 2588.40 | 2610.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 2608.35 | 2588.40 | 2610.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 2628.45 | 2596.41 | 2612.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:00:00 | 2628.45 | 2596.41 | 2612.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 2623.55 | 2601.84 | 2613.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:30:00 | 2624.90 | 2601.84 | 2613.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 2589.60 | 2599.39 | 2611.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 2603.75 | 2599.39 | 2611.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2592.25 | 2590.84 | 2603.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 2602.90 | 2590.84 | 2603.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 2598.80 | 2592.43 | 2603.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 2595.00 | 2592.43 | 2603.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 2592.35 | 2592.42 | 2602.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:30:00 | 2600.50 | 2592.42 | 2602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2554.40 | 2525.74 | 2540.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 2554.40 | 2525.74 | 2540.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2527.25 | 2526.04 | 2539.14 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 2589.30 | 2547.66 | 2546.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 2602.00 | 2558.53 | 2551.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2713.85 | 2731.36 | 2693.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 2713.85 | 2731.36 | 2693.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 2735.65 | 2738.33 | 2716.56 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 2703.70 | 2719.09 | 2720.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 2700.00 | 2715.28 | 2718.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 2712.70 | 2709.63 | 2714.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:45:00 | 2711.35 | 2709.63 | 2714.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 2705.60 | 2708.82 | 2713.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 2702.40 | 2709.76 | 2713.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 2720.00 | 2711.81 | 2714.01 | SL hit (close>static) qty=1.00 sl=2719.95 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 2704.00 | 2697.22 | 2696.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 2722.20 | 2702.22 | 2698.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 2696.05 | 2700.98 | 2698.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 2696.05 | 2700.98 | 2698.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2686.05 | 2698.00 | 2697.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 2678.65 | 2698.00 | 2697.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 2683.50 | 2695.10 | 2696.20 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 2704.55 | 2695.76 | 2695.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 2740.70 | 2705.59 | 2699.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 2738.55 | 2744.98 | 2732.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:45:00 | 2741.10 | 2744.98 | 2732.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2704.35 | 2736.06 | 2730.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2704.35 | 2736.06 | 2730.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2692.85 | 2727.42 | 2727.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 2692.85 | 2727.42 | 2727.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 11:15:00 | 2704.35 | 2722.80 | 2725.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 2681.50 | 2714.54 | 2721.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 2704.20 | 2699.83 | 2708.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 2704.20 | 2699.83 | 2708.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 2695.00 | 2698.86 | 2707.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 14:15:00 | 2694.50 | 2698.86 | 2707.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:45:00 | 2693.40 | 2701.67 | 2706.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 2682.90 | 2701.67 | 2706.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 2692.70 | 2701.08 | 2703.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 2707.90 | 2702.44 | 2704.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 2717.35 | 2705.42 | 2705.46 | SL hit (close>static) qty=1.00 sl=2712.95 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 2723.00 | 2708.94 | 2707.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 2727.60 | 2712.67 | 2708.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 2692.00 | 2708.54 | 2707.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 2692.00 | 2708.54 | 2707.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 2680.30 | 2702.89 | 2704.92 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 2737.75 | 2707.42 | 2704.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 2761.80 | 2718.30 | 2710.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 2766.50 | 2776.74 | 2759.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:00:00 | 2766.50 | 2776.74 | 2759.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 2765.35 | 2774.46 | 2759.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 2765.35 | 2774.46 | 2759.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2739.50 | 2766.32 | 2758.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 2739.50 | 2766.32 | 2758.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2746.05 | 2762.26 | 2757.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 2745.80 | 2762.26 | 2757.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 2755.40 | 2760.26 | 2757.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 2756.90 | 2760.26 | 2757.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 2758.75 | 2759.96 | 2757.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 2774.00 | 2759.91 | 2757.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 10:15:00 | 2750.65 | 2758.06 | 2757.26 | SL hit (close<static) qty=1.00 sl=2754.65 alert=retest2 |

### Cycle 117 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 2744.65 | 2755.38 | 2756.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 2730.85 | 2750.47 | 2753.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2769.50 | 2743.95 | 2748.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 2753.35 | 2743.95 | 2748.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 2761.00 | 2747.36 | 2749.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 2749.80 | 2747.36 | 2749.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:15:00 | 2612.31 | 2644.40 | 2672.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 2624.65 | 2624.20 | 2652.57 | SL hit (close>ema200) qty=0.50 sl=2624.20 alert=retest2 |

### Cycle 118 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 2687.50 | 2659.39 | 2658.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 2700.85 | 2667.68 | 2662.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 2789.65 | 2793.91 | 2766.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 2785.55 | 2793.91 | 2766.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2773.60 | 2789.85 | 2766.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 2773.60 | 2789.85 | 2766.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 2777.00 | 2787.28 | 2767.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 2764.25 | 2787.28 | 2767.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 2779.90 | 2786.26 | 2770.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 2772.85 | 2786.26 | 2770.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2824.45 | 2799.43 | 2782.21 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 2754.95 | 2777.13 | 2778.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 2729.70 | 2749.10 | 2760.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 2735.80 | 2732.10 | 2744.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 2735.80 | 2732.10 | 2744.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 2737.55 | 2733.19 | 2743.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 2738.20 | 2733.19 | 2743.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 2745.00 | 2736.49 | 2743.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 2722.00 | 2736.49 | 2743.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2719.95 | 2733.18 | 2741.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 2706.25 | 2721.68 | 2728.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 2704.90 | 2713.67 | 2718.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 2711.00 | 2717.38 | 2718.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 2709.40 | 2716.88 | 2718.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 2717.45 | 2716.99 | 2718.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 2717.05 | 2716.99 | 2718.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-15 14:15:00 | 2738.55 | 2721.31 | 2719.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 2738.55 | 2721.31 | 2719.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 2757.50 | 2729.54 | 2723.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2748.85 | 2755.80 | 2743.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 2743.50 | 2755.80 | 2743.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 2731.60 | 2753.54 | 2744.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 2731.60 | 2753.54 | 2744.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 2720.75 | 2746.99 | 2742.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 2720.75 | 2746.99 | 2742.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 2688.80 | 2735.35 | 2737.84 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 2759.55 | 2738.65 | 2736.10 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 2732.15 | 2735.02 | 2735.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 2716.60 | 2730.07 | 2732.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2687.00 | 2658.29 | 2676.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 2687.00 | 2658.29 | 2676.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 2691.15 | 2664.87 | 2677.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 2691.15 | 2664.87 | 2677.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 2674.90 | 2666.87 | 2677.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 2663.70 | 2666.87 | 2677.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 2668.55 | 2666.62 | 2675.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 2675.00 | 2647.42 | 2647.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 2675.00 | 2647.42 | 2647.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 2679.30 | 2661.32 | 2654.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 2667.30 | 2679.97 | 2670.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 2667.30 | 2679.97 | 2670.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 2678.75 | 2679.73 | 2671.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 2660.00 | 2679.73 | 2671.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2675.10 | 2678.80 | 2671.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 2656.95 | 2678.80 | 2671.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2684.50 | 2679.94 | 2672.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:45:00 | 2679.50 | 2679.94 | 2672.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 2674.85 | 2680.11 | 2674.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 2674.85 | 2680.11 | 2674.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 2680.95 | 2680.28 | 2674.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 2697.10 | 2683.64 | 2676.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:30:00 | 2692.90 | 2688.91 | 2681.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 2651.25 | 2681.38 | 2678.47 | SL hit (close<static) qty=1.00 sl=2672.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2609.00 | 2666.90 | 2672.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 2592.60 | 2642.46 | 2659.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 2628.00 | 2619.89 | 2641.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:00:00 | 2628.00 | 2619.89 | 2641.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 2651.25 | 2624.42 | 2634.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 2651.25 | 2624.42 | 2634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 2655.65 | 2630.67 | 2636.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 2652.10 | 2630.67 | 2636.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 2622.30 | 2631.77 | 2636.33 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 2648.90 | 2640.12 | 2639.46 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2583.90 | 2630.25 | 2635.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2575.00 | 2619.20 | 2629.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2530.00 | 2529.26 | 2551.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:00:00 | 2530.00 | 2529.26 | 2551.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 2530.00 | 2510.41 | 2522.02 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 2531.70 | 2521.57 | 2521.55 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 2508.10 | 2518.88 | 2520.32 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 2530.00 | 2522.99 | 2522.06 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 2510.05 | 2521.11 | 2521.55 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 2537.00 | 2524.48 | 2522.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2548.50 | 2529.28 | 2525.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 2502.20 | 2526.05 | 2524.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 2502.20 | 2526.05 | 2524.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 2500.85 | 2521.01 | 2522.50 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 12:15:00 | 2553.65 | 2528.86 | 2525.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 2581.40 | 2541.27 | 2532.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 2612.85 | 2614.78 | 2592.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 2612.85 | 2614.78 | 2592.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2607.45 | 2613.10 | 2602.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 2604.85 | 2613.10 | 2602.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 2601.40 | 2610.76 | 2602.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 2602.20 | 2610.76 | 2602.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 2598.00 | 2608.21 | 2602.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 2601.00 | 2608.21 | 2602.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 2627.85 | 2612.14 | 2604.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 2632.30 | 2614.76 | 2606.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 2589.60 | 2609.20 | 2607.27 | SL hit (close<static) qty=1.00 sl=2597.95 alert=retest2 |

### Cycle 135 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 2586.00 | 2604.56 | 2605.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 2562.60 | 2596.17 | 2601.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2593.65 | 2591.04 | 2597.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 2599.95 | 2591.04 | 2597.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2605.75 | 2593.98 | 2598.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:45:00 | 2607.80 | 2593.98 | 2598.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2606.70 | 2596.53 | 2598.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 2606.70 | 2596.53 | 2598.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 2611.15 | 2599.45 | 2599.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 2618.30 | 2599.45 | 2599.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 2617.00 | 2602.96 | 2601.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 2657.35 | 2614.15 | 2607.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 2712.00 | 2713.83 | 2695.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:15:00 | 2703.00 | 2713.83 | 2695.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2690.10 | 2709.08 | 2695.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 2690.10 | 2709.08 | 2695.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2683.05 | 2703.88 | 2694.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 2680.35 | 2703.88 | 2694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2713.80 | 2705.86 | 2696.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 2669.65 | 2705.86 | 2696.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 2717.20 | 2711.74 | 2701.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 2683.20 | 2711.74 | 2701.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 2705.55 | 2709.30 | 2702.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 2705.55 | 2709.30 | 2702.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 2701.65 | 2707.77 | 2702.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:15:00 | 2699.90 | 2707.77 | 2702.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 2710.35 | 2708.29 | 2703.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:30:00 | 2718.95 | 2708.63 | 2704.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 2677.55 | 2700.69 | 2701.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 2677.55 | 2700.69 | 2701.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 2675.10 | 2688.66 | 2694.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2689.15 | 2664.88 | 2674.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 2660.90 | 2673.05 | 2675.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:00:00 | 2660.90 | 2664.04 | 2670.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:30:00 | 2643.00 | 2661.68 | 2668.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:45:00 | 2655.75 | 2662.08 | 2667.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 2656.95 | 2661.05 | 2666.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 2647.95 | 2661.05 | 2666.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 2669.50 | 2655.75 | 2662.12 | SL hit (close>static) qty=1.00 sl=2666.90 alert=retest2 |

### Cycle 138 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 2685.80 | 2666.78 | 2666.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 2697.40 | 2672.90 | 2669.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 2675.60 | 2679.51 | 2673.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 11:00:00 | 2675.60 | 2679.51 | 2673.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 2676.20 | 2678.85 | 2673.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 2687.40 | 2681.66 | 2676.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 2645.10 | 2674.08 | 2673.94 | SL hit (close<static) qty=1.00 sl=2672.25 alert=retest2 |

### Cycle 139 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 2653.70 | 2670.00 | 2672.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2629.75 | 2661.95 | 2668.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 2523.10 | 2516.27 | 2544.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 2523.10 | 2516.27 | 2544.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2545.00 | 2522.01 | 2544.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 2545.00 | 2522.01 | 2544.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 2524.10 | 2522.43 | 2542.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:15:00 | 2520.55 | 2522.43 | 2542.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 2520.00 | 2521.11 | 2536.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:00:00 | 2519.70 | 2521.67 | 2533.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 2521.65 | 2521.66 | 2532.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 2529.30 | 2518.77 | 2527.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 2496.05 | 2511.81 | 2522.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 2497.15 | 2497.56 | 2507.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 2494.15 | 2497.05 | 2506.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 2489.40 | 2460.37 | 2458.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 2489.40 | 2460.37 | 2458.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 2502.90 | 2468.88 | 2462.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2510.90 | 2516.49 | 2501.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2510.90 | 2516.49 | 2501.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2510.90 | 2515.13 | 2503.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2503.40 | 2515.13 | 2503.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2471.15 | 2506.34 | 2500.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2471.15 | 2506.34 | 2500.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2475.35 | 2500.14 | 2498.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2456.80 | 2500.14 | 2498.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2462.65 | 2490.30 | 2493.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 2462.40 | 2484.72 | 2491.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2488.75 | 2481.27 | 2488.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 2488.75 | 2481.27 | 2488.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 2477.45 | 2480.51 | 2487.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 2478.70 | 2480.51 | 2487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2475.65 | 2479.54 | 2486.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 2477.45 | 2479.54 | 2486.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2297.60 | 2307.90 | 2329.22 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 2352.20 | 2333.54 | 2332.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 2359.00 | 2345.68 | 2339.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 2384.55 | 2392.96 | 2379.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 2384.55 | 2392.96 | 2379.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2389.45 | 2392.26 | 2380.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 2384.20 | 2392.26 | 2380.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 2386.45 | 2391.10 | 2381.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:45:00 | 2394.50 | 2384.61 | 2381.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 2395.35 | 2385.68 | 2382.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 2425.15 | 2444.75 | 2447.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 2425.15 | 2444.75 | 2447.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 2413.85 | 2438.57 | 2444.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2438.75 | 2434.84 | 2441.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 2438.75 | 2434.84 | 2441.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2450.95 | 2438.06 | 2442.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 2450.95 | 2438.06 | 2442.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 2447.40 | 2439.93 | 2442.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:15:00 | 2450.15 | 2439.93 | 2442.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 2449.15 | 2441.77 | 2443.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:15:00 | 2453.00 | 2441.77 | 2443.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 2460.00 | 2445.42 | 2444.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2469.65 | 2452.54 | 2448.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2487.90 | 2495.01 | 2482.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 2509.50 | 2496.80 | 2484.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 2444.40 | 2486.03 | 2481.89 | SL hit (close<static) qty=1.00 sl=2482.50 alert=retest2 |

### Cycle 145 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 2438.50 | 2476.53 | 2477.94 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 2480.35 | 2468.31 | 2467.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 2485.00 | 2471.64 | 2468.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2481.15 | 2484.27 | 2478.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 14:30:00 | 2504.45 | 2491.00 | 2484.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 2504.40 | 2493.70 | 2486.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 2513.00 | 2494.37 | 2488.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 2514.05 | 2496.25 | 2490.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 2488.20 | 2494.64 | 2490.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:45:00 | 2480.95 | 2494.64 | 2490.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 2485.20 | 2492.75 | 2489.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 2464.75 | 2492.75 | 2489.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 2460.95 | 2486.39 | 2487.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2460.95 | 2486.39 | 2487.09 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 2530.95 | 2485.29 | 2484.74 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 2442.55 | 2483.32 | 2486.15 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 2513.00 | 2486.80 | 2485.23 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 2471.95 | 2488.05 | 2488.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 2465.35 | 2483.51 | 2486.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 2449.70 | 2443.61 | 2459.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 2462.55 | 2443.61 | 2459.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 2450.30 | 2444.95 | 2458.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 2463.90 | 2444.95 | 2458.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 2451.30 | 2446.22 | 2458.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:30:00 | 2449.60 | 2446.22 | 2458.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 2455.30 | 2448.03 | 2457.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 2457.60 | 2448.03 | 2457.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2465.10 | 2451.45 | 2458.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:30:00 | 2468.15 | 2451.45 | 2458.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 2460.00 | 2453.16 | 2458.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 2423.05 | 2453.16 | 2458.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2438.85 | 2450.30 | 2456.91 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 2465.60 | 2455.83 | 2455.13 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 2447.10 | 2455.98 | 2456.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2405.50 | 2445.02 | 2451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 11:15:00 | 2443.95 | 2441.55 | 2448.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 12:00:00 | 2443.95 | 2441.55 | 2448.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 2435.00 | 2440.24 | 2447.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:30:00 | 2423.40 | 2437.73 | 2445.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 2427.80 | 2436.98 | 2444.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 15:15:00 | 2417.15 | 2436.98 | 2444.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 14:15:00 | 2306.41 | 2329.40 | 2350.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 2302.23 | 2328.49 | 2348.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 09:15:00 | 2349.95 | 2332.78 | 2348.67 | SL hit (close>ema200) qty=0.50 sl=2332.78 alert=retest2 |

### Cycle 154 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2375.10 | 2355.98 | 2355.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2384.90 | 2370.35 | 2363.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 10:15:00 | 2379.95 | 2384.10 | 2374.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 2379.95 | 2384.10 | 2374.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 2394.60 | 2385.65 | 2378.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 10:15:00 | 2396.50 | 2385.67 | 2379.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 11:45:00 | 2398.70 | 2390.83 | 2383.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:45:00 | 2396.80 | 2392.27 | 2385.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 2400.95 | 2391.46 | 2386.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2414.95 | 2396.16 | 2388.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 2417.35 | 2403.18 | 2396.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 2386.20 | 2395.64 | 2395.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 2386.20 | 2395.64 | 2395.70 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 11:15:00 | 2407.40 | 2396.23 | 2395.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 2410.65 | 2400.35 | 2397.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2390.90 | 2402.59 | 2399.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2390.90 | 2402.59 | 2399.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 2378.35 | 2397.74 | 2397.84 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 2407.00 | 2399.59 | 2398.67 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 2389.10 | 2398.17 | 2398.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 2375.75 | 2390.91 | 2394.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2384.80 | 2380.72 | 2387.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 2396.90 | 2380.72 | 2387.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 2377.30 | 2380.03 | 2386.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 2384.85 | 2380.03 | 2386.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 2392.95 | 2382.62 | 2387.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 2392.95 | 2382.62 | 2387.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 2386.95 | 2383.48 | 2387.24 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 2400.00 | 2391.12 | 2390.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2421.00 | 2397.09 | 2392.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 2461.85 | 2464.05 | 2448.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 2474.20 | 2464.05 | 2448.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2466.30 | 2480.17 | 2465.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2466.30 | 2480.17 | 2465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2478.65 | 2479.87 | 2466.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2489.70 | 2479.87 | 2466.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 2510.45 | 2605.24 | 2617.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2510.45 | 2605.24 | 2617.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 2485.00 | 2567.15 | 2597.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2581.50 | 2538.90 | 2567.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:45:00 | 2567.00 | 2538.90 | 2567.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 2582.30 | 2547.58 | 2568.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 2560.95 | 2551.75 | 2568.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:30:00 | 2543.95 | 2566.67 | 2571.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 2558.60 | 2565.58 | 2570.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 2563.50 | 2565.17 | 2569.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2645.00 | 2580.67 | 2575.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2645.00 | 2580.67 | 2575.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 2652.90 | 2602.10 | 2586.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2697.50 | 2698.09 | 2665.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2697.50 | 2698.09 | 2665.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 2740.00 | 2751.51 | 2745.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 2743.20 | 2751.51 | 2745.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2726.00 | 2746.41 | 2743.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2726.00 | 2746.41 | 2743.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 2705.10 | 2738.15 | 2739.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 11:15:00 | 2692.60 | 2729.04 | 2735.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 2716.70 | 2708.07 | 2719.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 2716.70 | 2708.07 | 2719.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 2732.50 | 2712.96 | 2720.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:45:00 | 2733.80 | 2712.96 | 2720.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 2728.60 | 2716.09 | 2721.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 2723.40 | 2716.09 | 2721.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:30:00 | 2722.50 | 2717.71 | 2721.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:00:00 | 2718.70 | 2711.24 | 2715.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 2739.10 | 2716.81 | 2717.95 | SL hit (close>static) qty=1.00 sl=2734.60 alert=retest2 |

### Cycle 164 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 2729.20 | 2719.29 | 2718.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 2769.00 | 2729.23 | 2723.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2742.60 | 2748.22 | 2738.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 2745.60 | 2748.22 | 2738.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2729.50 | 2744.48 | 2737.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 2729.50 | 2744.48 | 2737.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 2725.00 | 2740.58 | 2736.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:30:00 | 2731.40 | 2740.58 | 2736.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 2737.10 | 2738.47 | 2736.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 2736.30 | 2738.47 | 2736.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2742.30 | 2739.23 | 2737.15 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 2722.10 | 2736.88 | 2737.18 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 2767.50 | 2743.00 | 2739.93 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 2710.90 | 2736.50 | 2737.51 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2757.40 | 2736.80 | 2735.71 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 2727.90 | 2740.25 | 2741.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 2722.00 | 2736.60 | 2739.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 14:15:00 | 2703.20 | 2696.21 | 2707.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 2703.20 | 2696.21 | 2707.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2680.60 | 2693.08 | 2705.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 2664.20 | 2693.08 | 2705.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 2719.80 | 2665.76 | 2678.82 | SL hit (close>static) qty=1.00 sl=2709.60 alert=retest2 |

### Cycle 170 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2739.00 | 2691.70 | 2689.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 2775.00 | 2744.20 | 2730.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 2776.10 | 2783.43 | 2759.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 2776.10 | 2783.43 | 2759.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2788.60 | 2795.39 | 2776.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 2781.00 | 2795.39 | 2776.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 2776.40 | 2791.59 | 2776.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 2776.40 | 2791.59 | 2776.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 2777.00 | 2788.68 | 2776.68 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 2724.60 | 2767.00 | 2769.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2718.10 | 2735.96 | 2749.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2735.90 | 2732.91 | 2745.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:30:00 | 2732.40 | 2732.91 | 2745.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2743.50 | 2735.03 | 2745.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2746.60 | 2735.03 | 2745.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2727.40 | 2733.51 | 2743.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 2724.20 | 2733.51 | 2743.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 2721.40 | 2730.82 | 2741.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 2720.90 | 2689.99 | 2705.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2587.99 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2585.33 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:15:00 | 2584.86 | 2608.44 | 2632.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 2579.00 | 2578.51 | 2596.00 | SL hit (close>ema200) qty=0.50 sl=2578.51 alert=retest2 |

### Cycle 172 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 2585.30 | 2556.79 | 2553.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2609.50 | 2576.30 | 2568.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 2710.90 | 2711.60 | 2683.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 2678.40 | 2704.73 | 2689.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2678.40 | 2704.73 | 2689.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2678.40 | 2704.73 | 2689.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2687.20 | 2701.22 | 2689.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:45:00 | 2689.00 | 2701.22 | 2689.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2684.30 | 2697.84 | 2688.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2672.30 | 2697.84 | 2688.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2672.90 | 2692.85 | 2687.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 2678.60 | 2688.04 | 2685.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 12:15:00 | 2667.80 | 2681.91 | 2683.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 2667.80 | 2681.91 | 2683.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 2660.30 | 2675.36 | 2679.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2696.90 | 2674.43 | 2677.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 2696.90 | 2674.43 | 2677.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 2704.00 | 2680.34 | 2679.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 2710.90 | 2686.45 | 2682.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2691.30 | 2692.02 | 2686.54 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2668.60 | 2684.37 | 2684.89 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2687.10 | 2685.19 | 2685.04 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 2677.00 | 2683.55 | 2684.31 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 2693.00 | 2685.16 | 2684.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 13:15:00 | 2703.00 | 2689.76 | 2686.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 2681.30 | 2688.86 | 2686.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 2706.10 | 2688.86 | 2686.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 2713.90 | 2705.18 | 2699.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 2836.00 | 2846.96 | 2847.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 2836.00 | 2846.96 | 2847.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 2829.90 | 2842.55 | 2845.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2807.50 | 2807.04 | 2821.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:30:00 | 2805.80 | 2807.04 | 2821.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2802.60 | 2805.49 | 2817.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 2802.60 | 2805.49 | 2817.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2811.00 | 2793.65 | 2804.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2811.00 | 2793.65 | 2804.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2816.00 | 2798.12 | 2805.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2816.00 | 2798.12 | 2805.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2816.90 | 2808.58 | 2808.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 2828.20 | 2808.58 | 2808.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 2823.80 | 2811.62 | 2810.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2824.60 | 2814.69 | 2811.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2811.20 | 2814.46 | 2812.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 2811.20 | 2814.46 | 2812.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2810.00 | 2813.57 | 2812.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 2803.90 | 2813.57 | 2812.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 2797.20 | 2810.30 | 2810.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 2788.10 | 2800.02 | 2804.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2792.20 | 2778.52 | 2786.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 2792.20 | 2778.52 | 2786.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2782.00 | 2779.22 | 2786.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 2772.70 | 2779.22 | 2786.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 2787.90 | 2784.98 | 2784.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2787.90 | 2784.98 | 2784.71 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 2779.60 | 2783.91 | 2784.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 2774.40 | 2781.38 | 2782.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2764.10 | 2759.44 | 2767.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 2764.10 | 2759.44 | 2767.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 2771.30 | 2761.81 | 2768.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 2770.90 | 2761.81 | 2768.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 2768.70 | 2763.19 | 2768.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 2760.00 | 2763.19 | 2768.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2740.00 | 2722.20 | 2719.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 2740.00 | 2722.20 | 2719.79 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2714.60 | 2721.54 | 2722.11 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 2742.40 | 2723.61 | 2722.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2786.90 | 2739.74 | 2730.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 2760.60 | 2762.70 | 2747.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 2760.60 | 2762.70 | 2747.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2729.30 | 2755.43 | 2746.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 2759.60 | 2752.71 | 2747.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 2760.00 | 2753.33 | 2748.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:00:00 | 2761.00 | 2755.93 | 2750.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 2740.80 | 2747.12 | 2747.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 2740.80 | 2747.12 | 2747.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 2717.90 | 2741.28 | 2745.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2765.50 | 2744.57 | 2745.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 2762.20 | 2744.57 | 2745.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 2766.90 | 2749.04 | 2747.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 2774.10 | 2757.63 | 2752.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2784.50 | 2790.72 | 2779.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 2784.50 | 2790.72 | 2779.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 2764.10 | 2785.40 | 2777.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 2764.10 | 2785.40 | 2777.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2762.90 | 2780.90 | 2776.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 2762.90 | 2780.90 | 2776.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 2767.00 | 2776.41 | 2774.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 2767.00 | 2776.41 | 2774.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 2767.60 | 2774.64 | 2774.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 2765.00 | 2774.64 | 2774.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 2764.00 | 2772.52 | 2773.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2731.30 | 2762.97 | 2768.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2755.60 | 2714.09 | 2728.42 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2753.80 | 2737.93 | 2736.31 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2716.30 | 2734.28 | 2735.32 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2750.00 | 2738.03 | 2736.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2845.00 | 2770.55 | 2754.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 2829.50 | 2829.68 | 2799.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 2831.50 | 2829.68 | 2799.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 2809.40 | 2822.69 | 2803.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 2809.40 | 2822.69 | 2803.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2823.40 | 2823.83 | 2810.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 2811.40 | 2823.83 | 2810.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2845.90 | 2869.15 | 2855.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 2848.70 | 2869.15 | 2855.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 2821.00 | 2859.52 | 2851.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 2821.00 | 2859.52 | 2851.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 2818.60 | 2845.17 | 2846.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 2813.80 | 2835.05 | 2841.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 2817.60 | 2816.06 | 2827.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 2817.60 | 2816.06 | 2827.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 2805.70 | 2807.62 | 2818.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 2812.30 | 2807.62 | 2818.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 2791.00 | 2804.95 | 2814.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:30:00 | 2808.80 | 2804.95 | 2814.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2789.50 | 2783.30 | 2790.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 2789.50 | 2783.30 | 2790.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 2785.90 | 2783.95 | 2789.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 2785.20 | 2783.95 | 2789.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2805.10 | 2788.18 | 2791.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 2805.10 | 2788.18 | 2791.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 2804.00 | 2791.34 | 2792.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 2805.00 | 2791.34 | 2792.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 2816.80 | 2796.43 | 2794.55 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 2780.90 | 2794.69 | 2795.37 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 2802.10 | 2796.27 | 2795.71 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 2779.50 | 2792.92 | 2794.24 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 2841.70 | 2798.77 | 2795.89 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2798.30 | 2807.92 | 2808.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 2791.20 | 2802.94 | 2806.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2804.60 | 2803.27 | 2805.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 2811.80 | 2803.27 | 2805.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2786.90 | 2800.00 | 2804.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 2780.30 | 2796.22 | 2801.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 2773.30 | 2792.32 | 2798.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 2782.50 | 2788.54 | 2795.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 2782.00 | 2787.10 | 2793.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2799.40 | 2789.41 | 2793.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 2802.60 | 2789.41 | 2793.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 2796.10 | 2790.75 | 2794.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2794.30 | 2790.75 | 2794.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 2813.60 | 2796.80 | 2796.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2813.60 | 2796.80 | 2796.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 2817.70 | 2805.65 | 2801.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 2801.70 | 2804.89 | 2802.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 2801.70 | 2804.89 | 2802.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 2801.40 | 2804.19 | 2802.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2812.90 | 2804.19 | 2802.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 2825.60 | 2810.64 | 2805.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 2833.90 | 2865.50 | 2866.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 2833.90 | 2865.50 | 2866.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 2815.00 | 2855.40 | 2861.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2837.70 | 2829.14 | 2840.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 2837.70 | 2829.14 | 2840.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2818.20 | 2820.23 | 2831.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 2829.60 | 2820.23 | 2831.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 2766.10 | 2789.54 | 2808.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 2746.10 | 2789.54 | 2808.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 2752.90 | 2770.63 | 2794.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:45:00 | 2753.60 | 2757.21 | 2775.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 2752.20 | 2756.23 | 2770.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2753.70 | 2754.73 | 2767.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 2734.10 | 2757.06 | 2762.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2787.20 | 2762.58 | 2763.17 | SL hit (close>static) qty=1.00 sl=2768.90 alert=retest2 |

### Cycle 202 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 2788.70 | 2767.80 | 2765.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 2792.00 | 2772.64 | 2767.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 2778.60 | 2782.04 | 2774.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 2778.60 | 2782.04 | 2774.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 2775.80 | 2780.80 | 2774.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 2773.30 | 2780.80 | 2774.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2793.10 | 2783.26 | 2776.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 2799.30 | 2783.26 | 2776.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 2797.10 | 2790.45 | 2780.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 2771.30 | 2800.28 | 2798.58 | SL hit (close<static) qty=1.00 sl=2773.30 alert=retest2 |

### Cycle 203 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 2772.60 | 2794.75 | 2796.21 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 2798.10 | 2792.90 | 2792.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 2811.70 | 2796.66 | 2794.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2801.70 | 2808.25 | 2801.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 2802.40 | 2808.25 | 2801.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 2808.70 | 2808.34 | 2802.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 2813.60 | 2808.34 | 2802.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 2812.00 | 2808.63 | 2803.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2788.70 | 2805.18 | 2802.64 | SL hit (close<static) qty=1.00 sl=2797.60 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 2792.80 | 2801.04 | 2801.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2775.90 | 2792.05 | 2796.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2793.70 | 2783.06 | 2788.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2793.10 | 2783.06 | 2788.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2795.90 | 2785.63 | 2789.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 2802.20 | 2785.63 | 2789.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 2820.00 | 2796.39 | 2793.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 2841.70 | 2813.27 | 2803.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 2848.80 | 2857.78 | 2839.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 2848.80 | 2857.78 | 2839.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2834.20 | 2853.06 | 2838.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 2835.00 | 2853.06 | 2838.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2832.80 | 2849.01 | 2838.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 2841.00 | 2849.01 | 2838.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 2844.80 | 2859.28 | 2860.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 2844.80 | 2859.28 | 2860.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 2824.90 | 2849.69 | 2855.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2866.60 | 2851.00 | 2854.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 2866.60 | 2851.00 | 2854.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2871.50 | 2855.10 | 2855.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:15:00 | 2876.00 | 2855.10 | 2855.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 2892.30 | 2862.54 | 2859.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 2922.50 | 2874.53 | 2865.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2934.30 | 2947.47 | 2930.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 2930.70 | 2947.47 | 2930.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2925.20 | 2945.30 | 2938.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 2925.20 | 2945.30 | 2938.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2917.90 | 2939.82 | 2936.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 2917.90 | 2939.82 | 2936.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 2925.80 | 2933.53 | 2933.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 2914.60 | 2929.74 | 2932.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 2905.00 | 2903.65 | 2913.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 2893.00 | 2903.65 | 2913.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2773.00 | 2737.31 | 2762.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 2773.00 | 2737.31 | 2762.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2777.30 | 2745.31 | 2763.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 2777.50 | 2745.31 | 2763.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2762.00 | 2757.79 | 2764.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 2760.10 | 2757.79 | 2764.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2753.60 | 2756.95 | 2763.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 2749.10 | 2754.60 | 2761.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2780.90 | 2767.60 | 2765.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2780.90 | 2767.60 | 2765.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 2798.90 | 2787.83 | 2782.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2770.40 | 2784.34 | 2781.34 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 2764.20 | 2776.34 | 2777.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 2755.00 | 2769.05 | 2773.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 2763.40 | 2762.62 | 2769.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 2763.40 | 2762.62 | 2769.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 2760.30 | 2762.15 | 2768.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 2756.30 | 2760.98 | 2767.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 2739.20 | 2720.01 | 2717.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 2739.20 | 2720.01 | 2717.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 2744.80 | 2724.97 | 2720.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 2727.00 | 2728.54 | 2724.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 2722.40 | 2728.54 | 2724.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 2724.80 | 2727.80 | 2724.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 2724.60 | 2727.80 | 2724.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2740.70 | 2733.19 | 2727.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 2732.30 | 2733.19 | 2727.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2741.60 | 2737.61 | 2732.95 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 2727.00 | 2730.87 | 2731.06 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 2734.20 | 2731.54 | 2731.35 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 2728.70 | 2730.97 | 2731.10 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 2735.10 | 2731.80 | 2731.44 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 2716.00 | 2729.79 | 2730.73 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 2736.90 | 2729.93 | 2729.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 2742.70 | 2732.49 | 2730.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 2745.90 | 2746.02 | 2739.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 2745.90 | 2746.02 | 2739.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2734.00 | 2745.79 | 2740.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:45:00 | 2735.50 | 2745.79 | 2740.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2743.50 | 2745.33 | 2740.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:15:00 | 2761.60 | 2742.61 | 2740.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 2757.90 | 2749.87 | 2744.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2767.80 | 2752.25 | 2747.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 2750.00 | 2757.40 | 2752.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 2746.40 | 2755.20 | 2751.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 2746.40 | 2755.20 | 2751.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2746.70 | 2753.50 | 2751.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 2744.60 | 2753.50 | 2751.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 2748.00 | 2752.40 | 2750.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 2753.10 | 2752.40 | 2750.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 2749.90 | 2753.94 | 2751.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 2786.20 | 2806.79 | 2807.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 2786.20 | 2806.79 | 2807.78 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 2817.00 | 2804.26 | 2803.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 2819.00 | 2807.20 | 2805.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2805.10 | 2806.78 | 2805.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2805.10 | 2806.78 | 2805.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2807.00 | 2806.83 | 2805.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 2806.90 | 2806.83 | 2805.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 2805.00 | 2806.46 | 2805.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 2825.50 | 2806.46 | 2805.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2817.00 | 2808.57 | 2806.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 2843.70 | 2811.96 | 2809.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 2839.30 | 2820.06 | 2813.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 2836.70 | 2819.26 | 2815.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 2833.80 | 2826.08 | 2820.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 2831.90 | 2833.65 | 2827.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 2830.50 | 2833.65 | 2827.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 2826.90 | 2832.30 | 2827.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 2826.50 | 2832.30 | 2827.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 2830.60 | 2831.96 | 2828.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 2831.20 | 2831.96 | 2828.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2822.90 | 2830.15 | 2827.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 2822.90 | 2830.15 | 2827.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2827.00 | 2829.52 | 2827.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 2827.00 | 2829.52 | 2827.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2828.60 | 2829.34 | 2827.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 2818.20 | 2825.87 | 2826.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 2818.20 | 2825.87 | 2826.25 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 2846.00 | 2828.93 | 2827.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 2854.50 | 2842.80 | 2837.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 11:15:00 | 2844.00 | 2845.09 | 2839.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 2844.00 | 2845.09 | 2839.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 2827.40 | 2841.91 | 2839.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 2825.80 | 2841.91 | 2839.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2833.00 | 2840.13 | 2838.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 2832.10 | 2840.13 | 2838.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 2832.10 | 2838.18 | 2838.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 2829.00 | 2836.35 | 2837.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 2836.00 | 2835.20 | 2836.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 2839.00 | 2835.20 | 2836.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 2852.70 | 2838.70 | 2838.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 2874.50 | 2847.94 | 2842.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 2867.00 | 2869.00 | 2859.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 2867.00 | 2869.00 | 2859.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2851.50 | 2865.50 | 2858.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2851.50 | 2865.50 | 2858.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2849.40 | 2862.28 | 2858.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 2849.40 | 2862.28 | 2858.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2852.40 | 2858.29 | 2856.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2850.20 | 2858.29 | 2856.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2863.70 | 2859.37 | 2857.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 2869.10 | 2861.32 | 2858.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 2865.30 | 2862.38 | 2859.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 2869.80 | 2862.38 | 2859.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 2848.70 | 2858.73 | 2858.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 2848.70 | 2858.73 | 2858.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 2840.10 | 2855.00 | 2857.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2780.50 | 2772.07 | 2789.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2777.90 | 2772.07 | 2789.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 2794.70 | 2776.60 | 2790.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 2794.70 | 2776.60 | 2790.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 2807.70 | 2782.82 | 2791.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 2818.90 | 2782.82 | 2791.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2797.50 | 2790.41 | 2793.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:45:00 | 2801.50 | 2790.41 | 2793.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2785.90 | 2789.50 | 2792.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 2774.90 | 2787.82 | 2791.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 2776.80 | 2787.82 | 2791.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 2815.00 | 2788.30 | 2789.48 | SL hit (close>static) qty=1.00 sl=2798.50 alert=retest2 |

### Cycle 226 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 2809.20 | 2792.48 | 2791.27 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 2784.90 | 2798.83 | 2799.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 2780.00 | 2795.06 | 2798.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 2743.00 | 2741.27 | 2761.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 2743.00 | 2741.27 | 2761.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2788.00 | 2748.71 | 2756.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 2788.00 | 2748.71 | 2756.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2775.00 | 2753.97 | 2758.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 2769.00 | 2760.24 | 2760.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 2766.70 | 2761.53 | 2761.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 2766.70 | 2761.53 | 2761.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 2792.20 | 2767.66 | 2764.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 2771.00 | 2775.88 | 2769.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 2770.00 | 2775.88 | 2769.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 2772.80 | 2775.26 | 2770.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 2775.20 | 2775.26 | 2770.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 2758.60 | 2771.93 | 2769.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 2753.40 | 2771.93 | 2769.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 2759.80 | 2769.50 | 2768.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 2759.80 | 2769.50 | 2768.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2814.50 | 2835.78 | 2823.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 2814.50 | 2835.78 | 2823.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2824.80 | 2833.58 | 2824.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 2830.00 | 2829.85 | 2824.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 2830.50 | 2831.61 | 2826.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 2832.30 | 2831.61 | 2826.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:45:00 | 2830.60 | 2830.27 | 2826.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2831.60 | 2830.54 | 2826.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 2817.90 | 2824.54 | 2824.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 2817.90 | 2824.54 | 2824.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2795.60 | 2815.25 | 2820.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 2759.90 | 2755.33 | 2777.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 2756.60 | 2755.33 | 2777.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2773.60 | 2761.76 | 2776.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2824.10 | 2761.76 | 2776.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2829.40 | 2775.29 | 2781.71 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2835.80 | 2787.39 | 2786.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2852.50 | 2813.63 | 2801.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2844.20 | 2851.02 | 2838.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 2844.20 | 2851.02 | 2838.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2841.90 | 2849.20 | 2839.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 2839.90 | 2849.20 | 2839.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 2828.90 | 2845.14 | 2838.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 2829.10 | 2845.14 | 2838.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2835.30 | 2843.17 | 2837.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2826.50 | 2843.17 | 2837.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 2845.80 | 2842.55 | 2838.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 2864.10 | 2842.55 | 2838.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 2904.30 | 2918.20 | 2919.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 2904.30 | 2918.20 | 2919.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 2886.30 | 2911.82 | 2916.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2910.40 | 2904.24 | 2909.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 2910.40 | 2904.24 | 2909.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 2907.00 | 2904.79 | 2909.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 2912.40 | 2904.79 | 2909.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 2911.80 | 2906.19 | 2909.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 2914.80 | 2906.19 | 2909.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2896.20 | 2904.19 | 2908.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 2891.00 | 2903.82 | 2906.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 2932.30 | 2908.13 | 2907.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 2932.30 | 2908.13 | 2907.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 2932.80 | 2913.06 | 2910.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2914.20 | 2924.13 | 2918.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2914.20 | 2924.13 | 2918.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 2892.50 | 2917.80 | 2915.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 2892.50 | 2917.80 | 2915.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 2891.20 | 2912.48 | 2913.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2867.60 | 2898.24 | 2906.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 2853.40 | 2852.85 | 2871.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:30:00 | 2865.20 | 2852.85 | 2871.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 2872.00 | 2859.27 | 2869.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 2876.30 | 2859.27 | 2869.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2867.40 | 2860.90 | 2868.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 2873.30 | 2860.90 | 2868.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 2865.00 | 2861.72 | 2868.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 2864.00 | 2861.72 | 2868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2862.00 | 2861.77 | 2867.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 2850.10 | 2861.58 | 2867.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 2879.30 | 2870.05 | 2869.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 2879.30 | 2870.05 | 2869.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2898.00 | 2875.64 | 2872.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 2878.20 | 2883.22 | 2877.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 2878.20 | 2883.22 | 2877.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 2889.60 | 2884.50 | 2878.21 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 2862.00 | 2874.20 | 2875.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2824.40 | 2860.97 | 2868.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2782.30 | 2781.26 | 2807.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:30:00 | 2783.40 | 2781.26 | 2807.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2717.10 | 2691.02 | 2717.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2717.10 | 2691.02 | 2717.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2717.00 | 2696.21 | 2717.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2722.80 | 2702.81 | 2718.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2741.70 | 2710.59 | 2720.57 | EMA400 retest candle locked (from downside) |

### Cycle 236 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2738.10 | 2728.00 | 2727.10 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 2714.00 | 2725.20 | 2725.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 2661.50 | 2711.31 | 2719.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2716.20 | 2688.49 | 2699.81 | EMA400 retest candle locked (from downside) |

### Cycle 238 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 2743.10 | 2708.55 | 2706.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 2745.00 | 2721.13 | 2713.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2725.00 | 2738.75 | 2728.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2725.00 | 2738.75 | 2728.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2725.30 | 2736.06 | 2728.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2680.70 | 2736.06 | 2728.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2691.30 | 2727.11 | 2724.97 | EMA400 retest candle locked (from upside) |

### Cycle 239 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 2699.90 | 2721.67 | 2722.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 2677.40 | 2700.41 | 2711.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 2632.90 | 2619.23 | 2653.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 2632.90 | 2619.23 | 2653.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2641.10 | 2627.33 | 2651.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 2635.00 | 2627.33 | 2651.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 2659.00 | 2633.55 | 2648.41 | SL hit (close>static) qty=1.00 sl=2653.50 alert=retest2 |

### Cycle 240 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 2679.80 | 2657.53 | 2656.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 2686.60 | 2663.34 | 2658.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2646.40 | 2696.59 | 2686.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 2648.50 | 2696.59 | 2686.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2638.80 | 2685.03 | 2682.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2638.80 | 2685.03 | 2682.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 241 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 2630.50 | 2674.12 | 2677.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 2622.20 | 2663.74 | 2672.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2649.20 | 2640.36 | 2656.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:15:00 | 2643.80 | 2640.36 | 2656.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2649.10 | 2642.31 | 2654.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 2627.80 | 2644.05 | 2653.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 2636.90 | 2587.43 | 2586.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 242 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2636.90 | 2587.43 | 2586.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 2651.60 | 2606.85 | 2596.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2623.10 | 2628.68 | 2611.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 2617.00 | 2628.68 | 2611.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2618.20 | 2625.34 | 2613.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 2608.90 | 2625.34 | 2613.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 2623.70 | 2625.01 | 2614.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 2642.10 | 2623.87 | 2615.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 2561.50 | 2614.32 | 2612.80 | SL hit (close<static) qty=1.00 sl=2613.00 alert=retest2 |

### Cycle 243 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 2560.90 | 2603.63 | 2608.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 2541.00 | 2576.05 | 2591.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2593.70 | 2579.58 | 2592.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 2582.10 | 2579.58 | 2592.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 2578.50 | 2579.36 | 2590.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 2580.90 | 2589.33 | 2593.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 2580.00 | 2590.02 | 2593.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 2580.00 | 2588.02 | 2591.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2525.90 | 2588.02 | 2591.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 2619.70 | 2576.03 | 2572.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 244 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2619.70 | 2576.03 | 2572.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2738.70 | 2631.71 | 2605.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 2741.30 | 2741.30 | 2709.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2766.50 | 2741.30 | 2709.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 2747.60 | 2744.96 | 2726.21 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2696.20 | 2737.08 | 2726.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2696.20 | 2737.08 | 2726.00 | SL hit (close<ema400) qty=1.00 sl=2726.00 alert=retest1 |

### Cycle 245 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 2710.90 | 2721.95 | 2722.61 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2750.30 | 2727.62 | 2725.12 | EMA200 above EMA400 |

### Cycle 247 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2699.90 | 2728.79 | 2730.81 | EMA200 below EMA400 |

### Cycle 248 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 2745.30 | 2727.00 | 2726.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 2756.00 | 2732.80 | 2729.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2763.90 | 2769.42 | 2757.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 2763.90 | 2769.42 | 2757.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 2774.00 | 2775.95 | 2766.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 2760.30 | 2775.95 | 2766.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2762.30 | 2773.22 | 2766.35 | EMA400 retest candle locked (from upside) |

### Cycle 249 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 2749.20 | 2762.44 | 2762.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 2730.90 | 2756.13 | 2759.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 2757.40 | 2752.84 | 2757.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 2760.00 | 2752.84 | 2757.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2752.80 | 2752.83 | 2757.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 2763.50 | 2752.83 | 2757.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 2750.00 | 2752.27 | 2756.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 2742.90 | 2748.81 | 2754.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2738.90 | 2748.81 | 2754.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 2742.30 | 2746.81 | 2752.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 2763.20 | 2749.64 | 2752.98 | SL hit (close>static) qty=1.00 sl=2756.60 alert=retest2 |

### Cycle 250 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 2785.00 | 2756.71 | 2755.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 2793.00 | 2767.60 | 2761.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 2796.80 | 2802.82 | 2786.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 2796.80 | 2802.82 | 2786.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 2784.10 | 2799.07 | 2786.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 2784.10 | 2799.07 | 2786.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2775.90 | 2794.44 | 2785.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 2793.20 | 2794.44 | 2785.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 2749.00 | 2793.97 | 2792.78 | SL hit (close<static) qty=1.00 sl=2772.00 alert=retest2 |

### Cycle 251 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 2760.60 | 2787.29 | 2789.86 | EMA200 below EMA400 |

### Cycle 252 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 2825.40 | 2794.58 | 2791.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 2841.70 | 2804.00 | 2795.82 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 10:15:00 | 1704.52 | 2023-05-22 14:15:00 | 1713.94 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-05-22 13:00:00 | 1704.37 | 2023-05-22 14:15:00 | 1713.94 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-06-16 09:15:00 | 1779.92 | 2023-06-19 11:15:00 | 1758.86 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2023-06-16 10:45:00 | 1775.39 | 2023-06-19 11:15:00 | 1758.86 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2023-06-16 13:30:00 | 1772.50 | 2023-06-19 11:15:00 | 1758.86 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-06-27 10:30:00 | 1714.93 | 2023-06-27 14:15:00 | 1726.44 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-07-05 09:15:00 | 1774.30 | 2023-07-07 11:15:00 | 1756.02 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-07-05 10:45:00 | 1770.46 | 2023-07-07 11:15:00 | 1756.02 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-07-05 12:15:00 | 1766.43 | 2023-07-07 11:15:00 | 1756.02 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-07-05 15:00:00 | 1771.51 | 2023-07-07 11:15:00 | 1756.02 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-07-14 09:15:00 | 1755.52 | 2023-07-14 09:15:00 | 1748.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-07-24 10:15:00 | 1831.47 | 2023-07-27 15:15:00 | 1811.80 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-07-25 09:15:00 | 1831.42 | 2023-07-27 15:15:00 | 1811.80 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-07-26 09:30:00 | 1840.58 | 2023-07-27 15:15:00 | 1811.80 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-07-26 11:15:00 | 1831.12 | 2023-07-27 15:15:00 | 1811.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-08-03 12:30:00 | 1790.53 | 2023-08-03 13:15:00 | 1817.77 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-08-14 09:15:00 | 1793.52 | 2023-08-21 14:15:00 | 1800.84 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-24 11:45:00 | 1787.09 | 2023-08-29 10:15:00 | 1786.20 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-08-24 12:15:00 | 1783.96 | 2023-08-29 10:15:00 | 1786.20 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2023-09-13 11:30:00 | 1927.94 | 2023-09-20 15:15:00 | 1933.56 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2023-09-13 12:15:00 | 1928.93 | 2023-09-20 15:15:00 | 1933.56 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2023-09-13 13:00:00 | 1923.35 | 2023-09-20 15:15:00 | 1933.56 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-09-13 14:00:00 | 1923.35 | 2023-09-20 15:15:00 | 1933.56 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2023-10-05 11:00:00 | 1885.70 | 2023-10-10 13:15:00 | 1903.68 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-10-09 09:15:00 | 1874.00 | 2023-10-10 13:15:00 | 1903.68 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-10-09 10:30:00 | 1886.00 | 2023-10-10 13:15:00 | 1903.68 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-10-09 11:00:00 | 1886.80 | 2023-10-10 13:15:00 | 1903.68 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1981.37 | 2023-10-17 11:15:00 | 1959.16 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-10-20 09:30:00 | 1956.22 | 2023-10-26 09:15:00 | 1858.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:30:00 | 1956.22 | 2023-10-27 11:15:00 | 1856.57 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2023-11-02 09:15:00 | 1882.71 | 2023-11-13 15:15:00 | 1914.39 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2023-11-20 12:30:00 | 1956.22 | 2023-11-23 13:15:00 | 1952.89 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-11-22 14:15:00 | 1960.16 | 2023-11-23 13:15:00 | 1952.89 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-11-28 13:00:00 | 1951.99 | 2023-11-28 14:15:00 | 1968.77 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-12-07 10:45:00 | 2054.18 | 2023-12-13 10:15:00 | 2041.88 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-12-07 12:30:00 | 2054.53 | 2023-12-13 10:15:00 | 2041.88 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-12-08 14:30:00 | 2052.79 | 2023-12-13 10:15:00 | 2041.88 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-12-19 11:15:00 | 2096.02 | 2023-12-20 13:15:00 | 2084.01 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-12-20 12:00:00 | 2106.97 | 2023-12-20 13:15:00 | 2084.01 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-12-22 10:45:00 | 2064.49 | 2023-12-26 15:15:00 | 2071.71 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2023-12-22 11:15:00 | 2064.79 | 2023-12-26 15:15:00 | 2071.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2023-12-22 12:00:00 | 2064.14 | 2023-12-26 15:15:00 | 2071.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-12-22 12:45:00 | 2058.42 | 2023-12-27 10:15:00 | 2086.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2023-12-26 12:15:00 | 2057.97 | 2023-12-27 10:15:00 | 2086.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-12-26 13:00:00 | 2055.58 | 2023-12-27 10:15:00 | 2086.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-12-26 13:30:00 | 2054.98 | 2023-12-27 10:15:00 | 2086.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-01-16 10:15:00 | 2108.55 | 2024-01-17 10:15:00 | 2076.65 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-01-16 14:00:00 | 2108.85 | 2024-01-17 10:15:00 | 2076.65 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-01-16 15:00:00 | 2114.80 | 2024-01-17 10:15:00 | 2076.65 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-01-19 14:45:00 | 2076.55 | 2024-01-19 15:15:00 | 2091.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-01-24 12:30:00 | 2053.60 | 2024-01-24 15:15:00 | 2080.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-02-02 09:15:00 | 2144.00 | 2024-02-05 11:15:00 | 2115.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-02-05 10:45:00 | 2130.25 | 2024-02-05 11:15:00 | 2115.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-02-08 10:30:00 | 2094.35 | 2024-02-09 11:15:00 | 2122.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-02-08 13:30:00 | 2094.20 | 2024-02-09 11:15:00 | 2122.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-02-22 12:15:00 | 2166.05 | 2024-02-22 12:15:00 | 2143.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-02-28 10:15:00 | 2214.00 | 2024-02-28 10:15:00 | 2189.95 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-03-05 11:15:00 | 2246.05 | 2024-03-06 10:15:00 | 2209.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-03-05 12:45:00 | 2248.45 | 2024-03-06 10:15:00 | 2209.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-03-05 14:30:00 | 2243.30 | 2024-03-06 10:15:00 | 2209.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-03-15 09:45:00 | 2170.75 | 2024-03-15 14:15:00 | 2197.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-03-15 10:15:00 | 2169.85 | 2024-03-15 14:15:00 | 2197.45 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-03-15 11:00:00 | 2177.30 | 2024-03-15 14:15:00 | 2197.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-03-15 11:45:00 | 2176.85 | 2024-03-15 14:15:00 | 2197.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-03-20 14:15:00 | 2181.55 | 2024-03-21 10:15:00 | 2198.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-03-21 09:45:00 | 2181.55 | 2024-03-21 10:15:00 | 2198.95 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-03-28 09:15:00 | 2223.60 | 2024-04-04 11:15:00 | 2268.05 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest1 | 2024-04-05 09:15:00 | 2270.55 | 2024-04-08 10:15:00 | 2264.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest1 | 2024-04-16 14:15:00 | 2239.25 | 2024-04-18 09:15:00 | 2264.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-13 10:30:00 | 2350.00 | 2024-05-13 15:15:00 | 2392.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-22 13:30:00 | 2437.40 | 2024-05-23 09:15:00 | 2404.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-06-03 11:45:00 | 2384.95 | 2024-06-04 11:15:00 | 2265.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:45:00 | 2384.95 | 2024-06-05 12:15:00 | 2262.20 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2024-06-24 10:30:00 | 2483.45 | 2024-07-01 14:15:00 | 2731.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 11:00:00 | 2487.60 | 2024-07-01 14:15:00 | 2736.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-16 09:15:00 | 2822.25 | 2024-07-18 11:15:00 | 2792.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-28 13:30:00 | 2702.40 | 2024-08-28 14:15:00 | 2720.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-08-28 15:15:00 | 2704.10 | 2024-09-02 09:15:00 | 2704.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-30 15:15:00 | 2696.95 | 2024-09-02 09:15:00 | 2704.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-09-09 14:15:00 | 2694.50 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-09-10 09:45:00 | 2693.40 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-09-10 10:15:00 | 2682.90 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-09-11 09:15:00 | 2692.70 | 2024-09-11 10:15:00 | 2717.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-09-18 10:00:00 | 2774.00 | 2024-09-18 10:15:00 | 2750.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-19 11:15:00 | 2749.80 | 2024-09-24 14:15:00 | 2612.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 11:15:00 | 2749.80 | 2024-09-25 11:15:00 | 2624.65 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2024-10-11 09:15:00 | 2706.25 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-14 10:45:00 | 2704.90 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-10-15 10:00:00 | 2711.00 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-15 13:00:00 | 2709.40 | 2024-10-15 14:15:00 | 2738.55 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-10-24 12:15:00 | 2663.70 | 2024-10-29 09:15:00 | 2675.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-24 13:45:00 | 2668.55 | 2024-10-29 09:15:00 | 2675.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-10-31 15:00:00 | 2697.10 | 2024-11-04 09:15:00 | 2651.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-11-01 18:30:00 | 2692.90 | 2024-11-04 09:15:00 | 2651.25 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-27 14:15:00 | 2632.30 | 2024-11-28 11:15:00 | 2589.60 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-12-06 13:30:00 | 2718.95 | 2024-12-09 09:15:00 | 2677.55 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-12 09:15:00 | 2660.90 | 2024-12-13 11:15:00 | 2669.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-12 12:00:00 | 2660.90 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-12 12:30:00 | 2643.00 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-12-12 14:45:00 | 2655.75 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-13 09:15:00 | 2647.95 | 2024-12-13 13:15:00 | 2685.80 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-12-16 15:00:00 | 2687.40 | 2024-12-17 09:15:00 | 2645.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-12-23 12:15:00 | 2520.55 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-12-23 15:15:00 | 2520.00 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2024-12-24 11:00:00 | 2519.70 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-12-24 12:00:00 | 2521.65 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-12-26 10:30:00 | 2496.05 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-27 10:45:00 | 2497.15 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-12-27 11:45:00 | 2494.15 | 2025-01-02 10:15:00 | 2489.40 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-01-22 14:45:00 | 2394.50 | 2025-01-28 13:15:00 | 2425.15 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-01-23 09:15:00 | 2395.35 | 2025-01-28 13:15:00 | 2425.15 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-02-01 10:45:00 | 2509.50 | 2025-02-01 12:15:00 | 2444.40 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-02-06 14:30:00 | 2504.45 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-02-07 09:30:00 | 2504.40 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-07 11:30:00 | 2513.00 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-02-07 13:30:00 | 2514.05 | 2025-02-10 09:15:00 | 2460.95 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-02-21 13:30:00 | 2423.40 | 2025-02-28 14:15:00 | 2306.41 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2025-02-21 14:45:00 | 2427.80 | 2025-02-28 15:15:00 | 2302.23 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-02-21 13:30:00 | 2423.40 | 2025-03-03 09:15:00 | 2349.95 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-02-21 14:45:00 | 2427.80 | 2025-03-03 09:15:00 | 2349.95 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-02-21 15:15:00 | 2417.15 | 2025-03-03 14:15:00 | 2375.10 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2025-03-06 10:15:00 | 2396.50 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-03-06 11:45:00 | 2398.70 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-03-06 14:45:00 | 2396.80 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-03-07 09:15:00 | 2400.95 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-03-10 09:15:00 | 2417.35 | 2025-03-10 15:15:00 | 2386.20 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-24 09:15:00 | 2489.70 | 2025-04-07 09:15:00 | 2510.45 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-04-08 11:30:00 | 2560.95 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-09 10:30:00 | 2543.95 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-04-09 12:30:00 | 2558.60 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-04-09 14:00:00 | 2563.50 | 2025-04-11 09:15:00 | 2645.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-04-24 13:15:00 | 2723.40 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-04-25 09:30:00 | 2722.50 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-04-25 14:00:00 | 2718.70 | 2025-04-25 14:15:00 | 2739.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-05-09 09:15:00 | 2664.20 | 2025-05-12 09:15:00 | 2719.80 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-21 12:15:00 | 2724.20 | 2025-05-28 11:15:00 | 2587.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2721.40 | 2025-05-28 11:15:00 | 2585.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 09:30:00 | 2720.90 | 2025-05-28 11:15:00 | 2584.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 12:15:00 | 2724.20 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2721.40 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2025-05-23 09:30:00 | 2720.90 | 2025-05-29 15:15:00 | 2579.00 | STOP_HIT | 0.50 | 5.22% |
| BUY | retest2 | 2025-06-13 10:30:00 | 2678.60 | 2025-06-13 12:15:00 | 2667.80 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-06-20 09:15:00 | 2706.10 | 2025-07-03 09:15:00 | 2836.00 | STOP_HIT | 1.00 | 4.80% |
| BUY | retest2 | 2025-06-23 10:30:00 | 2713.90 | 2025-07-03 09:15:00 | 2836.00 | STOP_HIT | 1.00 | 4.50% |
| SELL | retest2 | 2025-07-14 11:15:00 | 2772.70 | 2025-07-15 12:15:00 | 2787.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-17 15:15:00 | 2760.00 | 2025-07-28 10:15:00 | 2740.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-31 13:00:00 | 2759.60 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-31 15:15:00 | 2760.00 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-01 10:00:00 | 2761.00 | 2025-08-01 13:15:00 | 2740.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-10 14:15:00 | 2780.30 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-11 09:15:00 | 2773.30 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-11 11:30:00 | 2782.50 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-11 12:45:00 | 2782.00 | 2025-09-12 10:15:00 | 2813.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-16 09:15:00 | 2812.90 | 2025-09-23 09:15:00 | 2833.90 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2025-09-16 12:00:00 | 2825.60 | 2025-09-23 09:15:00 | 2833.90 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-09-26 10:15:00 | 2746.10 | 2025-10-01 13:15:00 | 2787.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-26 13:15:00 | 2752.90 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-29 11:45:00 | 2753.60 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-29 14:45:00 | 2752.20 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2734.10 | 2025-10-01 14:15:00 | 2788.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-03 15:15:00 | 2799.30 | 2025-10-08 10:15:00 | 2771.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-06 09:30:00 | 2797.10 | 2025-10-08 10:15:00 | 2771.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-10 14:15:00 | 2813.60 | 2025-10-13 09:15:00 | 2788.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-10 15:15:00 | 2812.00 | 2025-10-13 09:15:00 | 2788.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-17 15:15:00 | 2841.00 | 2025-10-24 11:15:00 | 2844.80 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-11-11 10:45:00 | 2749.10 | 2025-11-12 09:15:00 | 2780.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-11-19 13:30:00 | 2756.30 | 2025-11-26 13:15:00 | 2739.20 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-12-09 10:15:00 | 2761.60 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-12-09 11:45:00 | 2757.90 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-12-10 09:15:00 | 2767.80 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-12-10 13:15:00 | 2750.00 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-12-11 09:15:00 | 2753.10 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-12-11 09:45:00 | 2749.90 | 2025-12-16 15:15:00 | 2786.20 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-12-22 09:15:00 | 2843.70 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-22 10:45:00 | 2839.30 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-23 09:45:00 | 2836.70 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-23 14:15:00 | 2833.80 | 2025-12-26 13:15:00 | 2818.20 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-01-06 13:00:00 | 2869.10 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-06 14:45:00 | 2865.30 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-06 15:15:00 | 2869.80 | 2026-01-07 11:15:00 | 2848.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-13 12:45:00 | 2774.90 | 2026-01-14 10:15:00 | 2815.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2776.80 | 2026-01-14 10:15:00 | 2815.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-22 12:30:00 | 2769.00 | 2026-01-22 13:15:00 | 2766.70 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2026-01-29 13:30:00 | 2830.00 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-30 09:30:00 | 2830.50 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-30 10:15:00 | 2832.30 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-30 10:45:00 | 2830.60 | 2026-01-30 14:15:00 | 2817.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2864.10 | 2026-02-13 13:15:00 | 2904.30 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2026-02-17 15:15:00 | 2891.00 | 2026-02-18 10:15:00 | 2932.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-24 10:45:00 | 2850.10 | 2026-02-24 15:15:00 | 2879.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-16 12:15:00 | 2635.00 | 2026-03-16 14:15:00 | 2659.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-20 14:15:00 | 2627.80 | 2026-03-25 10:15:00 | 2636.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-03-27 15:15:00 | 2642.10 | 2026-03-30 09:15:00 | 2561.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-04-01 10:15:00 | 2582.10 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-01 11:00:00 | 2578.50 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-01 13:30:00 | 2580.90 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-01 15:15:00 | 2580.00 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2525.90 | 2026-04-06 14:15:00 | 2619.70 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest1 | 2026-04-10 09:15:00 | 2766.50 | 2026-04-13 09:15:00 | 2696.20 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2026-04-10 15:00:00 | 2747.60 | 2026-04-13 09:15:00 | 2696.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-04-13 10:30:00 | 2719.70 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2026-04-13 14:15:00 | 2717.00 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2718.90 | 2026-04-13 15:15:00 | 2710.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-04-24 13:30:00 | 2742.90 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-24 14:00:00 | 2738.90 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-24 14:45:00 | 2742.30 | 2026-04-27 09:15:00 | 2763.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-29 09:15:00 | 2793.20 | 2026-04-30 09:15:00 | 2749.00 | STOP_HIT | 1.00 | -1.58% |
