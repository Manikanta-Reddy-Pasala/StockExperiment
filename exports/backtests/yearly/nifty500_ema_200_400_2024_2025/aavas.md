# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1446.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 44
- **Target hits / Stop hits / Partials:** 5 / 44 / 6
- **Avg / median % per leg:** -0.62% / -1.88%
- **Sum % (uncompounded):** -34.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 2 | 9 | 0 | -1.48% | -16.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 2 | 9 | 0 | -1.48% | -16.2% |
| SELL (all) | 44 | 9 | 20.5% | 3 | 35 | 6 | -0.41% | -18.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 9 | 20.5% | 3 | 35 | 6 | -0.41% | -18.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 11 | 20.0% | 5 | 44 | 6 | -0.62% | -34.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 1682.95 | 1706.26 | 1706.32 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 1723.75 | 1706.44 | 1706.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 1728.45 | 1706.66 | 1706.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 11:15:00 | 1807.75 | 1808.93 | 1771.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:00:00 | 1807.75 | 1808.93 | 1771.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1777.85 | 1810.54 | 1775.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1777.85 | 1810.54 | 1775.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1777.80 | 1810.21 | 1775.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 1779.00 | 1810.21 | 1775.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1779.00 | 1809.90 | 1775.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1737.65 | 1809.90 | 1775.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1790.80 | 1809.71 | 1775.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1776.05 | 1809.71 | 1775.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1777.00 | 1809.21 | 1775.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 1777.00 | 1809.21 | 1775.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1778.85 | 1808.91 | 1775.69 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 1677.00 | 1755.95 | 1756.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 1671.05 | 1755.11 | 1755.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 1680.00 | 1679.59 | 1703.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 13:45:00 | 1680.00 | 1679.59 | 1703.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1690.95 | 1673.13 | 1693.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 1706.35 | 1673.13 | 1693.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1688.00 | 1673.28 | 1693.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 1677.80 | 1673.28 | 1693.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 1679.45 | 1673.47 | 1693.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:45:00 | 1677.25 | 1673.54 | 1693.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 1679.55 | 1672.46 | 1689.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1682.00 | 1672.37 | 1689.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 1677.15 | 1672.42 | 1689.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 1676.10 | 1672.65 | 1689.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:45:00 | 1676.00 | 1672.62 | 1688.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:45:00 | 1670.70 | 1669.63 | 1684.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1676.00 | 1670.03 | 1684.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 1708.75 | 1672.24 | 1684.34 | SL hit (close>static) qty=1.00 sl=1697.55 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 1723.70 | 1684.30 | 1684.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 1732.00 | 1690.23 | 1687.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 1674.40 | 1693.05 | 1688.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 1674.40 | 1693.05 | 1688.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1674.40 | 1693.05 | 1688.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 1672.60 | 1693.05 | 1688.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 1675.00 | 1692.87 | 1688.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 14:45:00 | 1682.25 | 1692.18 | 1688.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 1671.25 | 1691.85 | 1688.52 | SL hit (close<static) qty=1.00 sl=1672.20 alert=retest2 |

### Cycle 5 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1782.50 | 1867.47 | 1867.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1765.10 | 1861.58 | 1864.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 1849.00 | 1843.88 | 1854.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1849.00 | 1843.88 | 1854.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1844.00 | 1843.88 | 1854.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1807.00 | 1843.88 | 1854.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 1924.20 | 1835.41 | 1848.03 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1967.40 | 1855.13 | 1854.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1976.50 | 1856.34 | 1855.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1901.90 | 1903.71 | 1882.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 1901.90 | 1903.71 | 1882.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1883.00 | 1903.29 | 1882.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 1882.90 | 1903.29 | 1882.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1882.00 | 1903.08 | 1882.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1882.00 | 1903.08 | 1882.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1886.00 | 1902.91 | 1882.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1918.20 | 1902.91 | 1882.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1903.50 | 1882.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1981.60 | 1903.50 | 1882.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 1977.10 | 1904.18 | 1883.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 1977.10 | 1907.28 | 1885.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1974.90 | 1909.86 | 1887.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1891.00 | 1922.91 | 1899.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 1891.00 | 1922.91 | 1899.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1899.90 | 1922.68 | 1899.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1909.00 | 1922.39 | 1899.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 1901.90 | 1921.57 | 1899.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1903.80 | 1921.07 | 1900.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 1901.80 | 1919.71 | 1900.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1895.00 | 1919.47 | 1900.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1890.50 | 1919.47 | 1900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1900.30 | 1919.27 | 1900.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:15:00 | 1884.70 | 1919.27 | 1900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | SL hit (close<static) qty=1.00 sl=1884.40 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1741.10 | 1884.87 | 1885.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1714.00 | 1877.37 | 1881.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1649.00 | 1648.78 | 1716.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:45:00 | 1649.00 | 1648.78 | 1716.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1672.00 | 1646.47 | 1688.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 1674.30 | 1646.47 | 1688.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1675.10 | 1632.93 | 1667.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1675.10 | 1632.93 | 1667.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1667.40 | 1633.27 | 1667.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 1671.00 | 1633.27 | 1667.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1668.10 | 1633.62 | 1667.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1652.60 | 1640.14 | 1669.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 1658.40 | 1640.33 | 1669.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1654.90 | 1640.89 | 1668.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1651.80 | 1641.89 | 1668.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1665.70 | 1643.07 | 1668.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 1665.70 | 1643.07 | 1668.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1651.00 | 1643.37 | 1668.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1683.30 | 1644.79 | 1667.93 | SL hit (close>static) qty=1.00 sl=1674.90 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1377.40 | 1298.38 | 1298.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 1383.60 | 1299.22 | 1298.78 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 09:15:00 | 1644.55 | 2024-06-11 09:15:00 | 1809.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-12 11:15:00 | 1677.80 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-12-12 14:00:00 | 1679.45 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-12-12 14:45:00 | 1677.25 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1679.55 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-12-20 10:00:00 | 1677.15 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-20 13:00:00 | 1676.10 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-12-20 13:45:00 | 1676.00 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-12-31 09:45:00 | 1670.70 | 2025-01-03 09:15:00 | 1708.75 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-01-10 10:00:00 | 1661.15 | 2025-01-21 09:15:00 | 1701.60 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-01-10 11:00:00 | 1659.00 | 2025-01-21 09:15:00 | 1701.60 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-01-10 12:45:00 | 1660.45 | 2025-01-21 09:15:00 | 1701.60 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1646.85 | 2025-01-21 09:15:00 | 1701.60 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-02-01 14:30:00 | 1682.45 | 2025-02-03 09:15:00 | 1724.50 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-01 15:15:00 | 1681.00 | 2025-02-03 09:15:00 | 1724.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-02-11 14:45:00 | 1682.25 | 2025-02-12 09:15:00 | 1671.25 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-02-12 10:45:00 | 1679.20 | 2025-03-13 09:15:00 | 1847.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1807.00 | 2025-06-06 13:15:00 | 1924.20 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2025-06-12 14:00:00 | 1816.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1815.00 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-06-13 11:00:00 | 1816.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1837.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-06-16 12:00:00 | 1839.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1835.80 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-06-19 10:30:00 | 1844.10 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-23 13:15:00 | 1843.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-23 14:15:00 | 1843.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-24 10:00:00 | 1845.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-08 10:15:00 | 1981.60 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2025-07-08 11:15:00 | 1977.10 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-07-08 14:30:00 | 1977.10 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-07-09 12:45:00 | 1974.90 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2025-07-18 14:15:00 | 1909.00 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-07-21 11:00:00 | 1901.90 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-07-23 09:30:00 | 1903.80 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-07-23 15:00:00 | 1901.80 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-10-27 11:15:00 | 1652.60 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-27 12:15:00 | 1658.40 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-10-28 09:15:00 | 1654.90 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1651.80 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-03 15:00:00 | 1646.60 | 2025-11-07 09:15:00 | 1564.84 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1646.20 | 2025-11-10 09:15:00 | 1564.27 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-11-04 10:45:00 | 1647.20 | 2025-11-10 09:15:00 | 1563.89 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-03 15:00:00 | 1646.60 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1646.20 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.83% |
| SELL | retest2 | 2025-11-04 10:45:00 | 1647.20 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.77% |
| SELL | retest2 | 2025-11-04 15:00:00 | 1638.50 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-11-10 14:15:00 | 1636.90 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-11-11 12:30:00 | 1618.90 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1637.30 | 2025-11-28 11:15:00 | 1557.90 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1639.90 | 2025-11-28 12:15:00 | 1555.43 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-11-28 12:15:00 | 1553.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1637.30 | 2025-12-03 09:15:00 | 1475.91 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1639.90 | 2025-12-03 12:15:00 | 1473.57 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-12-03 12:15:00 | 1472.13 | TARGET_HIT | 0.50 | 10.00% |
