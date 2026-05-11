# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 1446.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 1 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 62 |
| PARTIAL | 7 |
| TARGET_HIT | 9 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 53
- **Target hits / Stop hits / Partials:** 9 / 53 / 7
- **Avg / median % per leg:** -0.26% / -1.91%
- **Sum % (uncompounded):** -17.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.55% | -21.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.55% | -21.8% |
| SELL (all) | 55 | 14 | 25.5% | 7 | 41 | 7 | 0.07% | 4.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 14 | 25.5% | 7 | 41 | 7 | 0.07% | 4.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 16 | 23.2% | 9 | 53 | 7 | -0.26% | -17.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 11:15:00 | 1581.25 | 1509.27 | 1509.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 13:15:00 | 1617.15 | 1517.78 | 1513.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 1562.05 | 1563.92 | 1544.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 10:00:00 | 1562.05 | 1563.92 | 1544.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 1655.85 | 1696.37 | 1652.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:45:00 | 1658.25 | 1695.94 | 1652.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 14:15:00 | 1662.00 | 1695.94 | 1652.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 09:15:00 | 1621.30 | 1694.40 | 1652.65 | SL hit (close<static) qty=1.00 sl=1650.20 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 1421.65 | 1625.69 | 1626.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 13:15:00 | 1413.85 | 1623.58 | 1625.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 11:15:00 | 1498.40 | 1497.51 | 1538.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-05 12:00:00 | 1498.40 | 1497.51 | 1538.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 1539.95 | 1499.01 | 1535.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:45:00 | 1537.05 | 1499.01 | 1535.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 1520.05 | 1499.22 | 1535.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:15:00 | 1510.40 | 1499.41 | 1535.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 09:15:00 | 1576.00 | 1501.39 | 1534.05 | SL hit (close>static) qty=1.00 sl=1541.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 09:15:00 | 1600.90 | 1541.08 | 1540.88 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 13:15:00 | 1489.40 | 1540.90 | 1541.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 14:15:00 | 1487.85 | 1540.38 | 1540.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 1469.00 | 1464.71 | 1490.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 10:00:00 | 1469.00 | 1464.71 | 1490.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1480.00 | 1465.01 | 1488.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 13:00:00 | 1471.00 | 1465.33 | 1488.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 14:45:00 | 1470.70 | 1465.53 | 1488.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 15:15:00 | 1469.00 | 1465.53 | 1488.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 15:00:00 | 1454.25 | 1465.16 | 1487.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-05 09:15:00 | 1323.90 | 1459.67 | 1482.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 09:15:00 | 1534.00 | 1455.05 | 1454.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 11:15:00 | 1550.00 | 1456.78 | 1455.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 1575.45 | 1578.60 | 1544.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 09:30:00 | 1577.70 | 1578.60 | 1544.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1592.50 | 1581.79 | 1550.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 1644.55 | 1577.44 | 1550.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-11 09:15:00 | 1809.01 | 1598.94 | 1564.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 1682.95 | 1706.26 | 1706.32 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-08-30 10:15:00)

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

### Cycle 8 — SELL (started 2024-10-24 09:15:00)

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

### Cycle 9 — BUY (started 2025-02-04 09:15:00)

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

### Cycle 10 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1782.50 | 1867.47 | 1867.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1765.10 | 1861.58 | 1864.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 1849.00 | 1843.88 | 1854.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1849.00 | 1843.88 | 1854.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1844.00 | 1843.88 | 1854.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1807.00 | 1843.88 | 1854.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 1924.20 | 1835.41 | 1848.03 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-27 09:15:00)

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

### Cycle 12 — SELL (started 2025-07-31 09:15:00)

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

### Cycle 13 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1377.40 | 1298.38 | 1298.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 1383.60 | 1299.22 | 1298.78 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-13 13:45:00 | 1658.25 | 2023-10-16 09:15:00 | 1621.30 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-10-13 14:15:00 | 1662.00 | 2023-10-16 09:15:00 | 1621.30 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2023-10-17 09:45:00 | 1662.05 | 2023-10-17 10:15:00 | 1648.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-12-08 12:15:00 | 1510.40 | 2023-12-12 09:15:00 | 1576.00 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2023-12-20 13:30:00 | 1509.25 | 2024-01-01 09:15:00 | 1542.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2023-12-29 10:00:00 | 1507.95 | 2024-01-01 09:15:00 | 1542.90 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-02-28 13:00:00 | 1471.00 | 2024-03-05 09:15:00 | 1323.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-28 14:45:00 | 1470.70 | 2024-03-05 09:15:00 | 1323.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-28 15:15:00 | 1469.00 | 2024-03-05 09:15:00 | 1322.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-29 15:00:00 | 1454.25 | 2024-03-05 09:15:00 | 1381.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-29 15:00:00 | 1454.25 | 2024-03-28 09:15:00 | 1308.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-02 09:15:00 | 1401.20 | 2024-04-04 09:15:00 | 1471.10 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2024-04-02 15:15:00 | 1408.00 | 2024-04-04 09:15:00 | 1471.10 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2024-04-03 14:15:00 | 1407.90 | 2024-04-04 09:15:00 | 1471.10 | STOP_HIT | 1.00 | -4.49% |
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
